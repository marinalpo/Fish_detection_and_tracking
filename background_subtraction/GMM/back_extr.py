import numpy as np
import argparse
import cv2


cap = cv2.VideoCapture('/Users/marinaalonsopoal/Documents/Telecos/Master/Research/Videos/Original/Andratx8_6L.MP4')
kernel = np.ones((50, 50), np.uint8)

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=100,nmixtures=5,backgroundRatio=0.7,noiseSigma=0)

while 1:
    ret, frame = cap.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ret, mask = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)

    frame[:, :, 0] = 0
    frame[:, :, 2] = 0

    cv2.imshow('mask', frame)


    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()