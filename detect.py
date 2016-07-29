# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
from numpy.linalg import inv
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
colors = []
# Next color available.
nextColor = 0

# Reset the colors available.
def resetColors():
	global nextColor
	nextColor = 0
	colors.append((0, 255, 0))
	colors.append((255, 0, 0))
	colors.append((0, 0, 255))
	colors.append((0, 0, 0))
	colors.append((100, 100, 100))

# Get the next available color:
def getColor():
	global nextColor
	if nextColor >= len(colors):
		resetColors()
	ret = colors[nextColor]
	nextColor = nextColor + 1
	return ret

# Class pedestrian. A pedestrian has position and color.
class Pedestrian:
	# Constructor:
	def __init__(self, x, y, color, initNumFrame):
		self.x = x
		self.y = y
		self.color = color
		self.frame = initNumFrame
		self.vx = 0
		self.vy = 0
	# Returns the color of the pedestrian:
	def getColor(self):
		return self.color
	# Returns the position of the pedestrian:
	def getPosition(self):
		return (self.x, self.y)
	# Returns the last frame when the pedestrian was sawed:
	def getNumFrame(self):
		return self.frame
	# Refresh the last frame when the pedestrian was sawed:
	def refreshNumFrame(currentNumFrame):
		self.frame = currentNumFrame
	# Get Pedestrian's velocity
 	def getVelocity(self):
		return (self.x, self.y)
	# Set new Pedestrian's state.
	def setState(self, x, y, vx, vy):
		self.x = x
		self.y = y
		self.vx = vx
		self.vy = vy

# Pedestrians recognized by the model:
pedestrians = []
# Check if 2 pedestrians are the same probably by their positions:
def checkSamePed(ped1, ped2):
	# threshold in pixels. If theirs middle points are 20 pixels close or less.
	threshold = 10
	(x1, y1) = ped1.getPosition()
	(x2, y2) = ped2.getPosition()
	dx = np.abs(x2-x1)
	dy = np.abs(y2-y1)
	if (dx < 20) and (dy < 20):
		return True
	else:
		return False

# Checks if a pedestrian is in the list:
def checkPedInList(ped1):
	for ped2 in pedestrians:
		if checkSamePed(ped1,ped2):
			return (True, ped2)
	return (False, None)

# Adds a pedestrian ir he/she is not in the list:
def addPedestrian(ped, currentNumFrame):
	(check, ped2) = checkPedInList(ped)
	if not check:
		pedestrians.append(ped)
	# If it is in the list we refresh his frame:
	else:
		ped2.refreshNumFrame(currentNumFrame)

# Remove a pedestrian of the list if he/she is not appearing more in the video:
def removeOldPed(currentNumFrame):
	# threshold for delete a pedestrian from the list: measured in frames.
	threshold = 20 #If the pedestrian does not appear in 20 frames is deleted from the list.
	for ped in pedestrians:
		lastNumFrameSawed = ped.getNumFrame()
		df = np.abs(lastNumFrameSawed-currenNumFrame)
		if df < threshold:
			pedestrians.remove(ped)

# getCentralPos returns the central point of a rectangle:
def getCentralPos(rect):
	for (x, y, w, h) in rect:
		return ((x+w)/2, (y+h)/2)


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi',fourcc, 60.0, (640,480))

#Global Variables
dt = 0.1
A = np.array([[1, dt],[0, 1]])
B = np.array([[dt**2/2.0],[dt]])
C = np.array([[1, 0]])
u = 1.5
Q = np.array([[0],[0]])
Pedestrian_estimate = Q
PedestrianAccel_noise_mag = 0.05
Our_noise_mag = 10
Ez = Our_noise_mag**2
Ex = PedestrianAccel_noise_mag**2 * np.array([[(dt**4)/4, (dt**3)/2], [(dt**3)/2, dt**2]])
P = Ex

#Kalmann filter frame_loc = [x, y], old_frame = obj;
def KalmannFilter(frame_loc, state):
    global Pedestrian_estimate, P
    Pedestrian_estimate = np.dot(A,Pedestrian_estimate) + B * u
    #Predict next covariance
    P = np.dot(np.dot(A, P), np.transpose(A)) + Ex
    #Kalman Gain
    K = np.dot(np.dot(P, np.transpose(C)), inv(np.dot(np.dot(C, P), np.transpose(C)) + Ez))
    #Update the state estimate.
    Pedestrian_estimate = Pedestrian_estimate + K * (frame_loc - np.dot(C,Pedestrian_estimate))
    return Pedestrian_estimate


# Loop over the frames over the video:
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

	# Reset the color available:
	resetColors()

	# draw the final bounding boxes
	if count == 0:
    	for (xA, yA, xB, yB) in pick:
			rectangleColor = getColor()
			caballero = Pedestrian((xA+xB)/2, (yA + yB)/2, rectangleColor, count)
			addPedestrian(caballero)		
			cv2.rectangle(image, (xA, yA), (xB, yB), rectangleColor, 2)
	else:

	#print (count)

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
