"""
CamPy: Python-based multi-camera recording software.
Integrates machine vision camera APIs with ffmpeg real-time compression.
Outputs one MP4 video file and metadata files for each camera

"campy" is the main console. 
User inputs are loaded from config yaml file using a command line interface (CLI) 
configurator parses the config arguments (and default params) into "params" dictionary.
configurator assigns params to each camera stream in the "cam_params" dictionary.
	* Camera index is set by "cameraSelection".
	* If param is string, it is applied to all cameras.
	* If param is list of strings, it is assigned to each camera, ordered by camera index.
Camera streams are acquired and encoded in parallel using multiprocessing.

Usage: 
campy-acquire ./configs/campy_config.yaml
"""

import os, time, sys, logging, threading, queue
from collections import deque
import numpy as np
import multiprocessing as mp
from campy import writer, display, configurator, process
from campy.trigger import trigger
from campy.cameras import unicam
from campy.utils.utils import HandleKeyboardInterrupt



def OpenSystems():
	# Configure parameters
	params = configurator.ConfigureParams()

	# Load Camera Systems and Devices
	#systems = unicam.LoadSystems(params)
	systems = {}
	#systems = unicam.GetDeviceList(systems, params)

	# Start camera triggers if configured
	#systems = trigger.StartTriggers(systems, params)

	return systems, params


def CloseSystems(systems, params):
	trigger.StopTriggers(systems, params)
	unicam.CloseSystems(systems, params)


def AcquireOneCamera(n_cam, frameQueue):
	# Initialize param dictionary for this camera stream
	cam_params = configurator.ConfigureCamParams(systems, params, n_cam)

	# Initialize queues for display, video writer, and stop messages
	writeQueue = deque()
	stopReadQueue = deque([],1)
	stopWriteQueue = deque([],1)

	# Start grabbing frames ("producer" thread)
	threading.Thread(
		target = unicam.GrabFrames,
daemon = True,
		args = (cam_params, writeQueue, frameQueue, stopReadQueue, stopWriteQueue,),
		).start()

	# Start video file writer (main "consumer" process)
	writer.WriteFrames(cam_params, writeQueue, stopReadQueue, stopWriteQueue)

def AcquireSimulation(n_cam, frameQueue):
	# Initialize param dictionary for this camera stream
	stop_event = mp.Event()
	cam_params = params

	# Initialize queues for display, video writer, and stop messages
	writeQueue = deque()
	stopReadQueue = deque([],1)
	stopWriteQueue = deque([],1)

	# Start grabbing frames ("producer" thread)
	t = threading.Thread(
		target = unicam.SimulateFrames,
daemon = False,
		args = (n_cam, writeQueue, frameQueue, stopReadQueue, stopWriteQueue, stop_event,),
		)
	
	t.start()

	try:
		t.join()  # or t.join(timeout) if you want to auto-exit eventually
	except KeyboardInterrupt:
		stop_event.set()
		t.join()

	print('Finishing AcquireSimulation')

	# Start video file writer (main "consumer" process)
	#writer.WriteFrames(cam_params, writeQueue, stopReadQueue, stopWriteQueue)

def Main():
	process_params = {
		'n_cams':params["numCams"],
		'model_path':'./models/mstride16_ostride2_filters8_costride2.single_instance.trt.FP16',
		'buffer_size':20,
		'num_keypoints':23,
		'img_shape': (3,1200,1920,3),
		'template': np.ones((20,23,3))
	}

	manager = mp.Manager()
	frame_queues = [manager.Queue(maxsize=10) for _ in range(params["numCams"])]
	
	with HandleKeyboardInterrupt():
		stop_event = mp.Event()
		processor = mp.Process(target=process.ProcessFrames, args=(process_params, frame_queues, stop_event,))
		processor.start()

		procs = []
		for i in range(params["numCams"]):
			acq_proc = mp.Process(target=AcquireSimulation, args=(i, frame_queues[i]))
			acq_proc.start()
			procs.append(acq_proc)

		for proc in procs:
			proc.join()

		# Acquire cameras in parallel with Windows- and Linux-compatible pool
		#p = mp.get_context("spawn").Pool(params["numCams"])
		#p.map_async(AcquireOneCamera,[(i, frame_queues[i]) for i in range(params["numCams"])]).get()
		#p.starmap_async(AcquireSimulation,[(i, frame_queues[i]) for i in range(params["numCams"])]).get()

	print('Camera acquisition finished')

	stop_event.set()  # signal FrameProcessor to stop
	print('Signaled stop event')
	processor.join()  # wait for it to exit

	#CloseSystems(systems, params)

# Open systems, creates global 'systems' and 'params' variables
systems, params = OpenSystems()