import numpy as np
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('video_path')
args = parser.parse_args()

cap = cv2.VideoCapture(args.video_path)
kernel = np.ones((50,50),np.uint8)

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=50,nmixtures=5,backgroundRatio=0.7,noiseSigma=0)

while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    cv2.imshow('opening',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()