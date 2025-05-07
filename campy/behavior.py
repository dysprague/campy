
import numpy as np
import scipy.io as sio

from numpy.lib.format import open_memmap
import cv2

from time import perf_counter
import traceback

import time

import serial
import os, csv

class Teensy:
    """
    Interfaces with the Teensy device over a serial port.
    The send_reward method sends a command to trigger tone and water reward.
    """
    def __init__(self, port):
        self.port = port
        try:
            self.conn = serial.Serial(port, 115200)
            self.conn.flushInput()
        except Exception as e:
            raise Exception(f"Unable to open serial port {port}: {e}")

    def send_reward(self):
        # Send command (e.g., "p") to trigger the reward pulse.
        self.conn.write(b'p')

    def send_trigger(self):
        # send command to trigger camera acquisition
        self.conn.write(b't')

    def __del__(self):
        if hasattr(self, 'conn') and self.conn.isOpen():
            self.conn.close()
        print(f"Closed serial port {self.port}")


class DataLogger:
    def __init__(self, filename, batch_size=500):
        self.filename = filename
        self.batch_size = batch_size
        # Open the file in line-buffered mode.
        self.file = open(filename, 'w', buffering=1)
        self.buffer = []
        self.file.write("frame_index,timestamp,raw_time,log_time,differences,thresholds,trigger\n")
        
    def log(self, frame_index, timestamp, raw_time, log_time, per_dim_diff, thresholds, triggered):
        diff_str = ",".join([f"{v:.3f}" for v in per_dim_diff])
        thresh_str = ",".join([f"{v:.3f}" for v in thresholds])
        line = f"{frame_index},{timestamp:.3f},{raw_time:.3f},{log_time:.3f},{diff_str},{thresh_str},{int(triggered)}\n"
        self.buffer.append(line)
        
        # When enough log entries are accumulated, write them all to disk.
        if len(self.buffer) >= self.batch_size:
            self.file.write("".join(self.buffer))
            self.file.flush()           # flush Python's internal buffer
            os.fsync(self.file.fileno()) # force flush to disk
            self.buffer = []  # clear the in-memory buffer
        
    def close(self):
        if self.buffer:
            self.file.write("".join(self.buffer))
            self.file.flush()
            os.fsync(self.file.fileno())
            self.buffer = []
        self.file.close()
        
    def __del__(self):
        self.close()

class BehaviorRecognizer:
    """
    Compares processed keypoint readings (already projected into D-dimensional space)
    against a candidate template (of shape (window_size, D)) using a sliding window.
    The per-dimension error is computed as the maximum absolute difference between the
    window's values and the candidate template.
    Returns:
      scalar_diff: overall maximum error (scalar),
      per_dim_diff: vector of maximum errors per dimension,
      data_arr: the buffered (processed) window.
    """
    def __init__(self, template, D):
         # candidate_template is provided from the configuration loader.
        self.template = template   # shape: (window_size, D)
        self.template_length = self.template.shape[0]
        self.D = D
        self.buffer = []

    def update(self, keypoint_reading):

        self.buffer.append(keypoint_reading.keypoints)

        # If we don't have enough samples yet, return None.
        if len(self.buffer) < self.template_length:
            return None
        
        # Keep only the most recent samples matching the template length.
        if len(self.buffer) > self.template_length:
            self.buffer = self.buffer[-self.template_length:]

        # Stack
        data_arr = np.array(self.buffer)  # shape: (template_length, n_features)

        # Compute similarity: max absolute difference
        diff = np.abs(data_arr - self.template)
        per_dim_diff = np.max(diff, axis=0)
        scalar_diff = np.max(per_dim_diff)
         
        return scalar_diff, per_dim_diff, data_arr


def normalize_skeleton(points_3d):

    SpineF = points_3d[3,:]  # shape: (n_frames, 3)
    SpineM = points_3d[4, :]  # shape: (n_frames, 3)

    rotangle = np.arctan2( -(SpineF[1] - SpineM[1]), (SpineF[0] - SpineM[0]) )

    global_rotmat = np.zeros((2, 2))

    global_rotmat[0, 0] = np.cos(rotangle)
    global_rotmat[0, 1] = -np.sin(rotangle)
    global_rotmat[1, 0] = np.sin(rotangle)
    global_rotmat[1, 1] = np.cos(rotangle) 

    markers_centered = points_3d - points_3d[4,:] #23x3

    markers_rotated = markers_centered 
    markers_rotated[:,:2] = np.transpose(global_rotmat @ np.transpose(markers_rotated[:,:2]))

    return markers_rotated

def undistort_points(points, K, dist_coeffs):
    """
    points: (N,2) array of pixel coordinates in one image
    K: (3,3) intrinsic matrix
    dist_coeffs: vector [k1,k2,p1,p2,(k3,…)] as OpenCV expects
    returns: (N,2) of normalized coordinates x',y' satisfying
             [x', y', 1]^T ∝ K^{-1} [u,v,1]^T after removing distortion
    """
    # OpenCV’s undistortPoints returns normalized coordinates if you omit P
    pts = points.reshape(-1,1,2).astype(np.float64)
    undist = cv2.undistortPoints(pts, K, dist_coeffs)  
    return undist.reshape(-1,2)

def build_projection_matrix(K, R, t):
    """Returns the 3×4 projection matrix P = K [R | t]."""
    return np.identity(3) @ np.hstack((R, t.reshape(3,1))) # Use identity for K because undistort already uses camera intrinsics

def triangulate_point_multiview(undist_points, P_list):
    """
    undist_points: list of (xi, yi) normalized coords, one per camera
    P_list:       list of corresponding 3×4 projection matrices
    returns:      X (3,) inhomogeneous 3D point
    """
    m = len(P_list)
    A = np.zeros((2*m, 4), dtype=np.float64)
    for i, ((x, y), P) in enumerate(zip(undist_points, P_list)):
        A[2*i    ] = x * P[2] - P[0]
        A[2*i + 1] = y * P[2] - P[1]

    # Solve A X = 0 via SVD:
    _, _, Vt = np.linalg.svd(A)
    X_homog = Vt[-1]        # last row of V^T
    X_homog /= X_homog[3]   # de-homogenize
    return X_homog[:3]


def triangulate(keypoints_2D, P_list, dist_coefs, K_list): #keys 3D is n_camsx23x2
    keypoints_2D[:,:,1] = 1200 - keypoints_2D[:,:,1] # Flip y vals
        
    undist_pts = np.zeros(keypoints_2D.shape) # n_cams x n_keypoints x 2

    points_3d = np.zeros((keypoints_2D.shape[1], 3))

    try:
        for i in range(keypoints_2D.shape[0]):
            undist_pts[i,:,:] = undistort_points(keypoints_2D[i,:,:], K_list[i], np.array(dist_coefs[i]))

        for j in range(keypoints_2D.shape[1]):
            uv_list = undist_pts[:,j,:] 
            points_3d[j,:] = triangulate_point_multiview(uv_list, P_list)

        return points_3d, undist_pts

    except Exception as e:
        #print(e)
        
        return points_3d, undist_pts

def correct_triangulations(points_3d, P_list, undist_pts, edges, bone_length_avg, w_bone=1.0):
    return points_3d #TODO: add triangulation corrections

def BehaviorData():

    behaviordata = {}

    behaviordata['frameNumber'] = []
    behaviordata['behaviorProcessTime'] = [] 
    behaviordata['finalTimeStamp'] = []

    return behaviordata

def SaveMetadata(vid_folder, behaviordata):

    csv_file = os.path.join(vid_folder, 'behavior_times.csv')

    keys_to_write = ['frameNumber', 'behaviorProcessTime', 'finalTimeStamp']
    length = len(behaviordata[keys_to_write[0]])

    with open(csv_file, 'w', newline='') as f:
        w = csv.writer(f)

        w.writerow(keys_to_write)
        for i in range(length):
            row = [behaviordata[key][i] for key in keys_to_write]

            w.writerow(row)


def ProcessBehavior(behavior_params, BehaviorQueue, stop_event):
    '''
    Args:

    Outputs:
        - Saves 3D keypoints to file
        - 
    '''
    print('Initializing behavior module')

    vid_folder = behavior_params["video_folder"]
    cam_calibration_path = behavior_params["calibration_path"]
    calibration_files = behavior_params["calibration_files"]
    skel_file = behavior_params["skeleton"]
    edge_lengths = behavior_params['edge_lengths']
    max_frames = behavior_params["numImagesToGrab"]
    n_cams = behavior_params['n_cams']
    save_path = behavior_params['save_path']

    skel_label = sio.loadmat(skel_file, simplify_cells=True)
    labels = skel_label['RP2']

    skeleton = skel_label['skeleton']
    # skeleton
    nodes = skeleton['joint_names']
    nodes = list(map(str, nodes))

    edges = skeleton['joints_idx']-1 # python indexing

    cam_extrinsics = []

    for file in calibration_files:
        params = sio.loadmat(f'{cam_calibration_path}/{file}', simplify_cells=True)
        cam_extrinsics.append({'K':params['K'], 'RDistort':params['RDistort'], 'TDistort':params['TDistort'], 'r':params['r'], 't':params['t']})

    P_list = []
    dist_coefs = []
    K_list = []

    for cam_vals in cam_extrinsics:
        K = np.transpose(cam_vals['K'])
        r = np.transpose(cam_vals['r'])
        t = -cam_vals['t']
        Rdist = cam_vals['RDistort']
        Tdist = cam_vals['TDistort']
        P_list.append(build_projection_matrix(K,r,t))
        K_list.append(K)
        dist_coefs.append([Rdist[0], Rdist[1], Tdist[0], Tdist[1]])


    #template_path = behavior_params["template_path"]
    #PCA_path = behavior_params["PCA_path"]

    #cam_calbration = sio.loadmat(cam_calibration_path, simplify_cells=True)
    #template = sio.loadmat(tempate_path, simplify_cells=True)
    #PCA_mat = sio.loadmat(PCA_path, simplify_cells=True)

    mm_peaks_and_vals = open_memmap(f'{save_path}/sleap_keys_2D.npy', mode='w+',
                           dtype=np.float64,
                           shape = (max_frames, n_cams, 23, 3)) # xy positions of keypoints from each camera and peak_val confidence levels
    mm_keys_3D = open_memmap(f'{save_path}/triang_keys_3D.npy', mode='w+',
                             dtype=np.float64,
                             shape = (max_frames, 23, 3))
    #mm_behav_pcs = open_memmap(f'{save_path}/behav_pcs.npy', mode='w+',
    #                        dtype=np.float64,
    #                        shape = (max_frames, 10))

    #num_PCs = template.shape[1]
    #template_length = template.shape[0]
    #BehavRec = BehaviorRecognizer(template, num_PCs)

    print("Behavior analysis module initialized and ready")

    frameNumber = 0

    behaviordata = BehaviorData()

    #logger = DataLogger(log_filename) # need to set
    #opcon_teensy = Teensy('/dev/ttyACM0') # need to set
    #cam_teensy = Teensy('/dev/ttyUSB') # need to set !!! - this might need to come earlier?

    start_time = perf_counter()

    print('Behavior initialized')

    while not stop_event.is_set():
        if not BehaviorQueue.empty():
            try:
                keypoints_2D, peak_vals  = BehaviorQueue.get() # 3x23x2, 3x23 matrices of peak locations and confidence

                keys_obtained = perf_counter()
                # Queue get is blocking if empty

                mm_peaks_and_vals[frameNumber, :,:,:2] = keypoints_2D 
                mm_peaks_and_vals[frameNumber, :,:,2] = peak_vals

                keypoints_3D, undist_pts = triangulate(keypoints_2D, P_list, dist_coefs, K_list)

                #TODO: add triangulation correction

                mm_keys_3D[frameNumber,:,:] = keypoints_3D

                points_rotated = normalize_skeleton(keypoints_3D)

                if (frameNumber%100) == 0:
                    mm_peaks_and_vals.flush() #Flush buffer of memory maps every 100 frames or so
                    mm_keys_3D.flush()

                frameNumber += 1

                # test to trigger reward every 100 frames
                #if frameNumber%100 == 0:
                #    print(f'Triggered reward on frame {frameNumber}')
                #    teensy.send_reward()

                
                #behav_PCA = process_keypoints(keypoints_3D, PCA_mat, num_PCs) # Produces num_pcs x1 vector
                #mm_behav_pcs[frameNumber,:] = behav_PCA # add to a mm

                # need to get 



                beh_processed = perf_counter()

                behaviordata['frameNumber'].append(frameNumber)
                behaviordata['behaviorProcessTime'].append(beh_processed-keys_obtained)
                behaviordata['finalTimeStamp'].append(beh_processed)


                #BehavRec.update(behav_PCA)

            except Exception as e:
                traceback.print_exc()

        else:
            time.sleep(0.005)

    SaveMetadata(vid_folder, behaviordata)

        
    

