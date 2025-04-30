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
from campy import writer, display, configurator, process, behavior
from campy.trigger import trigger
from campy.cameras import unicam
from campy.utils.utils import HandleKeyboardInterrupt



def OpenSystems():
	# Configure parameters
	params = configurator.ConfigureParams()

	# Load Camera Systems and Devices
	systems = unicam.LoadSystems(params)
	#systems = {}
	systems = unicam.GetDeviceList(systems, params)

	# Start camera triggers if configured
	systems = trigger.StartTriggers(systems, params)

	return systems, params


def CloseSystems(systems, params):
	trigger.StopTriggers(systems, params)
	unicam.CloseSystems(systems, params)


def AcquireOneCamera(n_cam, frameQueue, startQueue):
	# Initialize param dictionary for this camera stream
	print('Enter AcquireOneCamera', flush=True)
	cam_params = configurator.ConfigureCamParams(systems, params, n_cam)

	# Initialize queues for display, video writer, and stop messages
	writeQueue = queue.Queue(maxsize=100)
	stopReadQueue = deque([],1)
	stopWriteQueue = deque([],1)

	print('Start acquisition trigger')

	# Start grabbing frames ("producer" thread)
	t = threading.Thread(
		target = unicam.GrabFrames,
daemon = False,
		args = (cam_params, writeQueue, frameQueue, startQueue, stopReadQueue, stopWriteQueue,),
		)
	
	t.start()
	
	#try:
	#	t.join()
	#except KeyboardInterrupt:
	#	print('Keyboard interrupted acquisition')

	# Start video file writer (main "consumer" process)
	writer.WriteFrames(cam_params, writeQueue, stopReadQueue, stopWriteQueue)

def AcquireSimulation(n_cam, frameQueue, startQueue):
	# Initialize param dictionary for this camera stream
	stop_event = mp.Event()
	cam_params = params

	# Initialize queues for display, video writer, and stop messages
	writeQueue = queue.Queue(maxsize=100)
	stopReadQueue = deque([],1)
	stopWriteQueue = deque([],1)

	print(type(writeQueue))

	# Start grabbing frames ("producer" thread)
	threading.Thread(
		target = unicam.SimulateFrames,
daemon = True,
		args = (n_cam, writeQueue, frameQueue, startQueue, stopReadQueue, stopWriteQueue, stop_event,),
		).start()
	

	# Start video file writer (main "consumer" process)
	writer.WriteFrames(cam_params, writeQueue, stopReadQueue, stopWriteQueue)


	print('Finishing AcquireSimulation')

def Main():
	process_params = {
		'n_cams':params["numCams"],
		'model_path':'./models/250421_183045.single_instance.n=8280.trt.FP32',
		'buffer_size':20,
		'num_keypoints':23,
		'img_shape': (3,3,600,960),
		'template': np.ones((20,23,3))
	}

	behavior_params = {
		
	}

	manager = mp.Manager()
	frame_queues = [manager.Queue(maxsize=10) for _ in range(params["numCams"])]
	start_queues = [manager.Queue(maxsize=10) for _ in range(params["numCams"])]
	behavior_queue = manager.Queue(maxsize=10)
	
	with HandleKeyboardInterrupt():
		stop_event = mp.Event()
		#processor = mp.Process(target=process.ProcessFrames, args=(process_params, frame_queues, start_queues, stop_event,))
		#processor.start()

		behavior = mp.Process(target=behavior.ProcessBehavior, args=(behavior_params, behavior_queue, stop_event))
		behavior.start()

		# Acquire cameras in parallel with Windows- and Linux-compatible pool
		p = mp.get_context("spawn").Pool(params["numCams"])
		#p.map_async(AcquireOneCamera,[(i, frame_queues[i], start_queues[i]) for i in range(params["numCams"])]).get()
		p.starmap_async(AcquireOneCamera,[(i, frame_queues[i], start_queues[i]) for i in range(params["numCams"])]).get()

		print('Camera acquisition finished')

	stop_event.set()  # signal FrameProcessor to stop
	print('Signaled stop event')
	behavior.join()
	processor.join()  # wait for it to exit

	CloseSystems(systems, params)

# Open systems, creates global 'systems' and 'params' variables
systems, params = OpenSystems()


#procs = []

#print(params["numCams"])
#for i in range(params["numCams"]):
#	print('Cam')
#	acq_proc = mp.Process(target=AcquireOneCamera, args=(i, frame_queues[i], start_queues[i]))
#	acq_proc.start()
#	procs.append(acq_proc)

#for proc in procs:
#	proc.join()