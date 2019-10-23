import numpy as np
import cv2
import math
import sys
import os
import lane_detection_v2 as ld
from cache import Cache
import matplotlib.path as mpltPath

# Used to flip the frame upside down
# Needed because dashboard cam is mounted upside down
def invert_frame(frame):
	(H,W) = frame.shape[:2]
	center = (W/2, H/2)
	M = cv2.getRotationMatrix2D(center, 180, 1.0)
	frame = cv2.warpAffine(frame, M, (W,H))
	return frame

CONF = 0.2
position_camera = False
record_raw = False
record_processed = False
show_visuals = True
calib = False
invert = False
lanes = True
objects = True
detect_all = False
size = (640,360)
region_of_interest = [[0.44,0.65],[0.59,0.65],[.95,.95],[.1,.95]]	# The polygon region of interest on the forward roadway
roi_poly = mpltPath.Path(np.float32(region_of_interest)*np.float32(size))
pipe_roi = [region_of_interest[0],region_of_interest[1],region_of_interest[3], region_of_interest[2]]

#cap = cv2.VideoCapture('project_video.mp4')
cap = cv2.VideoCapture('tests/test_09/test_09_raw.mp4')
#cap = cv2.VideoCapture(cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

if record_raw:
	if not os.path.exists('tests/'+sys.argv[1]):
		os.makedirs('tests/'+sys.argv[1])
	out_raw = cv2.VideoWriter('tests/{}/{}_raw.mp4'.format(sys.argv[1],sys.argv[1]),cv2.VideoWriter_fourcc(*'XVID'), 30, (1280,720))
if record_processed:
	if not os.path.exists('tests/'+sys.argv[1]):
		os.makedirs('tests/'+sys.argv[1])
	out_processed = cv2.VideoWriter('tests/{}/{}_processed.mp4'.format(sys.argv[1],sys.argv[1]),cv2.VideoWriter_fourcc(*'XVID'), 30, (size[0]*2, size[1]))

lane_cache = Cache(max_size=2)	
# Cache stores previous lane curves so that when calculating new curves
# if there is a large variance then it is reduced by taking the mean
# with the cached lane curves

#Calibrate the camera
if calib:
	ld.calibrate(size=size)

# Init some variables for object detection
if objects:
	print("[INFO] loading model...")
	net = cv2.dnn.readNetFromCaffe("mobilenetssd/MobileNetSSD_deploy.prototxt.txt", "mobilenetssd/MobileNetSSD_deploy.caffemodel")

	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
				"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
				"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
				"sofa", "train", "tvmonitor"]

if position_camera:
	while(True):
		# ret = True/False if there is a next frame
		# frame = numpy pixel array
		ret, frame = cap.read()
		if not ret:
			break

		if invert:
			frame = invert_frame(frame)
		frame = cv2.resize(frame, size, interpolation = cv2.INTER_AREA)
		cv2.line(frame, (frame.shape[1]//2, 0), (frame.shape[1]//2, frame.shape[0]), (255,0,0), 1)
		cv2.polylines(frame, [np.array(np.float32(region_of_interest)*np.float32(size), np.int32)], True, (255,0,0), 1)

		cv2.imshow('Position Camera',frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break

print('\n===Running===')
while(True):
	# ret = True/False if there is a next frame
	# frame = numpy pixel array
	ret, frame = cap.read()
	if not ret:
		break

	if invert:
		frame = invert_frame(frame)

	if record_raw:
		out_raw.write(cv2.resize(frame, (1280, 720), interpolation = cv2.INTER_AREA))

	frame = cv2.resize(frame, size, interpolation = cv2.INTER_AREA)

	vehicles_detected = np.zeros_like(frame)

	if objects:
		# Grab the frame dimensions and convert it to a blob
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

		# Pass the blob through the network and obtain the detections and predictions
		net.setInput(blob)
		detections = net.forward()

		# Loop over the detections
		for i in np.arange(0, detections.shape[2]):
			# Extract the confidence (i.e., probability) associated with the prediction
			confidence = detections[0, 0, i, 2]

			# Filter out weak detections by ensuring the `confidence` is
			# Greater than the minimum confidence
			if confidence > CONF:
				# Extract the index of the class label from the
				# `detections`, then compute the (x, y)-coordinates of
				# the bounding box for the object
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# draw the prediction on the frame
				if detect_all or CLASSES[idx] in ['bus', 'car', 'motorbike']:

					W,H = (endX-startX), (endY-startY)

					object_dist = ((size[0] / W) * (size[1] / H))

					lane = ''

					midpoint = (startX+W//2,startY+H//2)

					color = (0,255,0)
					if roi_poly.contains_points([midpoint]) or (midpoint[0] > region_of_interest[0][0]*size[0] and midpoint[0] < region_of_interest[1][0]*size[0]):
						color = (255,0,0)
						lane = 'mine'
					elif midpoint[0] > size[0]//2:
						lane = 'right'
					else:
						lane = 'left'
					
					cv2.rectangle(vehicles_detected, midpoint, midpoint, color, 5)
					cv2.rectangle(vehicles_detected, (startX, startY), (endX, endY), color, 2)

					label_1 = '{} ({:.0f}%)'.format(CLASSES[idx], confidence*100)
					label_2 = 'dist: {:.0f}m'.format(object_dist)
					label_3 = 'lane: {}'.format(lane)

					y = startY - 5 if startY - 5 > 5 else endY + 5

					cv2.putText(vehicles_detected, label_1, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
					cv2.putText(vehicles_detected, label_2, (startX, y+H+15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
					cv2.putText(vehicles_detected, label_3, (startX, y+H+25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)

	#cv2.imshow('vehicles', vehicles_detected)
	if lanes:
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		detection_data = ld.vid_pipeline(rgb_frame, cache=lane_cache, roi=pipe_roi, show=show_visuals)

		processed_frame = np.maximum(detection_data[0],vehicles_detected)
		lane_curve = detection_data[1]
		left_curve = detection_data[2]
		right_curve = detection_data[3]
		vehicle_offset = detection_data[4]
		turn = detection_data[5]
		visuals = detection_data[6]

		print('\rLeft Curve: {:6.0f}\tRight Curve: {:6.0f}\tCenter Curve: {:6.0f}\tVehicle Offset: {:.4f}\t\tTurn: {}\t'.format(left_curve, right_curve, lane_curve, vehicle_offset, turn), end='')

		processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
		cv2.polylines(processed_frame, [np.array(np.float32(region_of_interest)*np.float32(size), np.int32)], True, (255,0,0), 1)

	if show_visuals and lanes:
		processed_frame = np.concatenate((visuals, processed_frame), axis=1)

	cv2.imshow('GCC',processed_frame)

	if record_processed:
		out_processed.write(processed_frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
if record_raw:
	out_raw.release()
if record_processed:
	out_processed.release()
cv2.destroyAllWindows()
