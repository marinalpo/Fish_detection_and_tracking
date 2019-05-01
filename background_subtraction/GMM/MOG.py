import numpy as np
import cv2

# MOG: Mixture of Gaussians

#cap = cv2.VideoCapture('/Users/marinaalonsopoal/Desktop/Video_test.mp4')
cap = cv2.VideoCapture('/Users/marinaalonsopoal/Documents/Telecos/Master/Research/Videos/Original/Andratx8_6L.MP4')

# PARAMETERS
hist = 200  # Length of the history
mixt = 5  # Number of Gaussian Mixtures (from 3 to 5)
ratio = 0.7  # Background Ratio
sigma = 0  # Noise strength (standard deviation of the brightness on each color channel). 0 means some automatic value.

kernel = np.ones((4,4), np.uint8)
kernel2 = np.ones((4, 4), np.uint8)

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=hist, nmixtures=mixt, backgroundRatio=ratio, noiseSigma=sigma)

while(1):
    ret, frame = cap.read()
    cv2.imshow('input', frame)
    frame = cv2.medianBlur(frame, 5)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    fgmask = fgbg.apply(frame)
    opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    cv2.imshow('frame', opening)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()