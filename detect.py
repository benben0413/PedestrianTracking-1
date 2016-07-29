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

# List of colors to identify pedestrians in the image:
colors = [(0, 255, 0),
		  (255, 0, 0),
		  (0, 0, 255),
		  (0, 0, 0),
		  (100, 100, 100),
		  (255, 0, 213),
		  (0, 213, 255),
		  (0, 114, 42)]
# True if it is available that color, false otherwise
available = []
for i in range(len(colors)):
	available.append(True)

# Returns the first available color.
def getColor():
	for (i, av) in enumerate(available):
		if av:
			return colors[i]
	return None

def setAvailable(color):
	global available
	for (i, cl) in enumerate(colors):
		if (cl == color):
			available[i] = True

def setNotAvailable(color):
	for (i, cl) in enumerate(colors):
		if (color==cl):
			available[i] = False

# # Next color available.
# nextColor = 0
#
# # Reset the colors available.
# def resetColors():
# 	global nextColor
# 	nextColor = 0
# 	colors.append((0, 255, 0))
# 	colors.append((255, 0, 0))
# 	colors.append((0, 0, 255))
# 	colors.append((0, 0, 0))
# 	colors.append((100, 100, 100))
#
# # Get the next available color:
# def getColor():
# 	global nextColor
# 	if nextColor >= len(colors):
# 		resetColors()
# 	ret = colors[nextColor]
# 	nextColor = nextColor + 1
# 	return ret

# Class pedestrian. A pedestrian has position and color.
class Pedestrian:
	# Constructor:
	def __init__(self, x, y, color, initNumFrame):
		self.x = x
		self.y = y
		self.color = color
		# Set available false
		setNotAvailable(color)
		self.frame = initNumFrame
	# Returns the color of the pedestrian:
	def getColor(self):
		return self.color
	# Set the current position of the pedestrian:
	def setPosition(self, x, y):
		self.x = x
		self.y = y
	# Returns the position of the pedestrian:
	def getPosition(self):
		return (self.x, self.y)
	# Returns the last frame when the pedestrian was seen:
	def getNumFrame(self):
		return self.frame
	# Refresh the last frame when the pedestrian was seen:
	def refreshNumFrame(self, currentNumFrame):
		self.frame = currentNumFrame

# Pedestrians recognized by the model:
pedestrians = []
# Check if 2 pedestrians are the same probably by their positions:
def checkSamePed(ped1, ped2):
	# threshold in pixels. If theirs middle points are 20 pixels close or less.
	thresholdX = 20
	thresholdY = 50
	(x1, y1) = ped1.getPosition()
	(x2, y2) = ped2.getPosition()
	dx = np.abs(x2-x1)
	dy = np.abs(y2-y1)
	if (dx < thresholdX) and (dy < thresholdY):
		return True
	else:
		return False

# Checks if a pedestrian is in the list: Returns a boolean and the
# pedestrian in the list. None if the pedestrian isn't in the list.
def checkPedInList(ped1):
	for ped2 in pedestrians:
		if checkSamePed(ped1,ped2):
			return (True, ped2)
	return (False, None)

# Adds a pedestrian if he/she is not in the list:
# If he is in the list is returned the old pedestrian with his color.
# If he is not in the list is returned himself.
def addPedestrian(ped, currentNumFrame):
	global pedestrians
	(check, ped2) = checkPedInList(ped)
	if not check:
		pedestrians.append(ped)
		return ped
	# If it is in the list we refresh his frame and position and set available
	# the color of ped.
	else:
		ped2.refreshNumFrame(currentNumFrame)
		(x, y) = ped.getPosition()
		ped2.setPosition(x, y)
		# Set available the color:
		setAvailable(ped.getColor())
		return ped2

# Remove a pedestrian of the list if he/she is not appearing more in the video:
def removeOldPed(currentNumFrame):
	global pedestrians
	# threshold for delete a pedestrian from the list: measured in frames.
	threshold = 10 #If the pedestrian does not appear in threshold frames is deleted from the list.
	for ped in pedestrians:
		lastNumFrameSawed = ped.getNumFrame()
		df = np.abs(lastNumFrameSawed-currentNumFrame)
		if df > threshold:
			# Set available his color
			setAvailable(ped.getColor())
			pedestrians.remove(ped)

# getCentralPos returns the central point of a rectangle:
def getCentralPos(xA,yA,xB,yB):
	return ((xA+xB)/2, (yA+yB)/2)


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi',fourcc, 60.0, (640,480))


# Loop over the frames over the video:
aux = 0
while True:

	# grab the current frame
	(grabbed, frame) = camera.read()

	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy
	image = imutils.resize(frame, width=min(640, frame.shape[1]))

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
		# Create a pedestrian:
		(x,y) = getCentralPos(xA,yA,xB,yB)
		color = getColor()
		ped = Pedestrian(x, y, color, count)
		# Add the pedestrian or check if he is in the list:
		ped2 = addPedestrian(ped, count)
		# Remove pedestrians that does not appear more
		removeOldPed(count)
		rectangleColor = ped2.getColor()

		print(ped.getColor(), ped.getPosition())
		print(ped2.getColor(), ped.getPosition())
		print("**")

		cv2.rectangle(image, (xA, yA), (xB, yB), rectangleColor, 2)

	print("------")

	# write the flipped frame
	if grabbed==True:
		writeimage = imutils.resize(image, width=640, height=480)
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
