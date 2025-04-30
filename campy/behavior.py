
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


def process_keypoints(BehavRec, markers, PCA_mat, num_PCs): #PCA matrix is 
    # --- Step 2: Egocentric Alignment ---
    # Define SpineF and SpineM indices according to MATLAB (1-indexed) to Python (0-indexed) mapping:
    #   SpineF = markers(:,:,4) -> Python index 3 -> double check these are the same
    #   SpineM = markers(:,:,5) -> Python index 4
    SpineF = markers[3, :]  # shape: (1, 3)
    SpineM = markers[4, :]  # shape: (1, 3)

    rotangle = np.arctan2( -(SpineF[:, 1] - SpineM[:, 1]), (SpineF[:, 0] - SpineM[:, 0]) )

    global_rotmat = np.zeros((2, 2))

    global_rotmatrix[0, 0] = np.cos(rotangle)
    global_rotmatrix[0, 1] = -np.sin(rotangle)
    global_rotmatrix[1, 0] = np.sin(rotangle)
    global_rotmatrix[1, 1] = np.cos(rotangle) 

    markers_centered = markers - markers[4,:] @

    markers_rotated = markers_aligned

    markers_rotated[] = global_rotmatrix.dot

    return keypoints_3D #TODO: return PCA decomposition of keypoint

def triangulate(keypoints_2D):

    return keypoints_2D #TODO: implement triangulation code based on camera calibrations



def ProcessBehavior(behavior_params, BehaviorQueue, stop_event):
    '''
    Args:

    Outputs:
        - Saves 3D keypoints to file
        - 
    '''
    print('Initializing behavior module')

    cam_calibration_path = behavior_params["calibration_path"]
    template_path = behavior_params["template_path"]
    PCA_path = behavior_params["PCA_path"]

    cam_calbration = sio.loadmat(cam_calibration_path, simplify_cells=True)
    template = sio.loadmat(tempate_path, simplify_cells=True)
    PCA_mat = sio.loadmat(PCA_path, simplify_cells=True)

    num_PCs = template.shape[1]

    BehavRec = BehaviorRecognizer(template, num_PCs)

    print("Behavior analysis module initialized and ready")

    frameNumber = 0

    while not stop_event.is_set():
        try:
            keypoints_2D, peak_vals  = BehaviorQueue.get() # 3x23x2, 3x23 matrices of peak locations and confidence
            # Queue get is blocking if empty

            keypoints_3D = triangulate(keypoints_2D, peak_vals, cam_calibration)

            behav_PCA = process_keypoints(keypoints_3D, PCA_mat, num_PCs) # Produces num_pcs x1 vector

            BehavRec.update(behav_PCA)

            #TODO: Save keypoints


        except Exception as e:
            traceback.print_exc()

        
    

