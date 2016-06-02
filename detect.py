# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import time
from collections import deque

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the video file")
args = vars(ap.parse_args())

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Counter que indica cuantos frames lleva el video.
count = 0
# Si no se indica el video se graba con la camara:
if not args.get("video", False):
	camera = cv2.VideoCapture(0)
else:
	camera = cv2.VideoCapture(args["video"])


# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*args["codec"])
out = cv2.VideoWriter('output.avi',-1, 20.0, (640,480))

# loop over the image paths

# Color del rectangulo que identifica una persona:
rectangleColor = (0, 255, 0)

# Loop over the frames over the video:
while True:

	# grab the current frame
	(grabbed, frame) = camera.read()
	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy
	image = imutils.resize(frame, width=min(400, frame.shape[1]))

	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
		padding=(8, 8), scale=1.05)

	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA, yA), (xB, yB), rectangleColor, 2)

	print (count)

	# write the flipped frame
	if grabbed==True:
		writeimage = cv2.flip(image,0)
		out.write(writeimage)

	# show the output frame
	cv2.imshow("Frame", image)
	count += 1
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

# Release everything if job is finished
camera.release()
out.release()
