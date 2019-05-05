import numpy as np
import cv2

cap = cv2.VideoCapture('/Users/marinaalonsopoal/Documents/Telecos/Master/Research/Videos/Original/Andratx8_6L.MP4')

# PARAMETERS
hist = 200  # Length of the history
mixt = 5  # Number of Gaussian Mixtures (from 3 to 5)
ratio = 0.7  # Background Ratio
sigma = 0  # Noise strength (standard deviation of the brightness on each color channel). 0 means some automatic value.

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=hist, nmixtures=mixt, backgroundRatio=ratio, noiseSigma=sigma)

num_frame = 0
capture = 1

while 1:
    num_frame = num_frame + 1
    ret, original_frame = cap.read()
    cv2.imshow('Original Video', original_frame)

    # Pre - processing
    frame = cv2.medianBlur(original_frame, 5)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Background Subtraction
    fgmask = fgbg.apply(frame)
    cv2.imshow('frame', fgmask)

    if capture == 1 and num_frame == 160:
        cv2.imwrite('/Users/marinaalonsopoal/Desktop/Original_Frame_' + str(num_frame) + '.jpg', original_frame)
        cv2.imwrite('/Users/marinaalonsopoal/Desktop/Back_Frame_' + str(num_frame) + '.jpg', fgmask)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()