import numpy as np
import cv2

alpha = 0.25  # Learning Factor [0,1] - Higher if background moves


class BackGroundSubtractor:

	def __init__(self, alpha, firstFrame):
		self.alpha = alpha
		self.backGroundModel = firstFrame

	def getForeground(self, frame):
		self.backGroundModel = frame * self.alpha + self.backGroundModel * (1 - self.alpha)
		return cv2.absdiff(self.backGroundModel.astype(np.uint8), frame)

# cam = cv2.VideoCapture('/Users/marinaalonsopoal/Desktop/Video_test.mp4')
cam = cv2.VideoCapture('/Users/marinaalonsopoal/Documents/Telecos/Master/Research/Videos/Original/Andratx8_6L.MP4')


def denoise(frame):
	frame = cv2.medianBlur(frame, 5)
	frame = cv2.GaussianBlur(frame, (5, 5), 0)
	return frame


ret, frame = cam.read()

if ret is True:
	backSubtractor = BackGroundSubtractor(alpha, denoise(frame))
	run = True
else:
	run = False

while run:
	ret, frame = cam.read()
	if ret is True:
		cv2.imshow('input',frame)
		foreGround = backSubtractor.getForeground(denoise(frame))
		ret, mask = cv2.threshold(foreGround, 15, 255, cv2.THRESH_BINARY)
		cv2.imshow('mask', mask)
		key = cv2.waitKey(10) & 0xFF
	else:
		break

	if key == 27:
		break

cam.release()
cv2.destroyAllWindows()