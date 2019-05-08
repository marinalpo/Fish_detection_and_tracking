import cv2
from processing import *

video = ['Andratx8_6L', 'Andratx9_6L', 'CalaEgos5L']
num_video = 2

dir_name = '/Users/marinaalonsopoal/Documents/Telecos/Master/Research/Videos/'+video[num_video]+'/'
cap = cv2.VideoCapture(dir_name + video[num_video] + '.mp4')

# Mixture of Gaussians parameters
hist = 200  # Length of the history
mixt = 5  # Number of Gaussian Mixtures (from 3 to 5)
ratio = 0.7  # Background Ratio
sigma = 0  # Noise strength (standard deviation of the brightness on each color channel). 0 means some automatic value.

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=hist, nmixtures=mixt, backgroundRatio=ratio, noiseSigma=sigma)

num_frame = 2
last_frame = 300

while num_frame <= last_frame:
    ret, original = cap.read()

    # Pre - processing
    frame = cv2.medianBlur(original, 5)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Background Subtraction
    back = fgbg.apply(frame)

    # Morphological Filters
    # hull = processBackground(back)

    # Save image files
    cv2.imwrite(dir_name + 'Originals/Original_Frame_' + str(num_frame) + '.jpg', original)
    cv2.imwrite(dir_name + 'Backgrounds/Background_Frame_' + str(num_frame) + '.jpg', back)
    # cv2.imwrite(dir_name + 'Hulls/Hull_Frame_' + str(num_frame) + '.jpg', hull)

    print('Processed Frame #' + str(num_frame))
    num_frame = num_frame + 1

cap.release()