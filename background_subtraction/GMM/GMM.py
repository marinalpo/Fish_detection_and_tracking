import cv2
from fishutils import *

# The videos that where used for this system where edited versions of the original ones, where the first minute of the
# video is removed to avoid camera settlement recordings, so the frame numbers do not match with the original ones.
video = ['Andratx8_6L', 'Andratx9_6L', 'CalaEgos5L']
num_video = 0  # Select number of video

# Directory from where the video will be extracted and where the processed frames will be saved
dir_name = '/Users/marinaalonsopoal/Documents/Telecos/Master/Research/Videos/'+video[num_video]+'/'
cap = cv2.VideoCapture(dir_name + video[num_video] + '.mp4')

# Mixture of Gaussians parameters
hist = 200  # Length of the history
mixt = 5  # Number of Gaussian Mixtures (from 3 to 5)
ratio = 0.7  # Background Ratio
sigma = 0  # Noise strength (standard deviation of the brightness on each color channel). 0 means some automatic value.

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=hist, nmixtures=mixt, backgroundRatio=ratio, noiseSigma=sigma)

capture = True
display = True
display_speed = [0, 20, 100]

num_frame = 1
initial_frame = 900
last_frame = 1000

while num_frame <= last_frame:
    ret, original = cap.read()

    # Display original video in real-time
    if display:
        cv2.putText(original, str(video[num_video]), (15, 45),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (173, 255, 47), 2)
        cv2.putText(original, 'Frame #' + str(num_frame), (15, 80),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Original Video', original)
        k = cv2.waitKey(display_speed[1]) & 0xff
        if k == 27:
            break

    # Pre - processing
    frame = cv2.medianBlur(original, 5)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Background Subtraction
    back = fgbg.apply(frame)

    if num_frame > initial_frame:
        # Morphological Filters
        hull = processBackground(back)

        # Save image files
        if capture:
            print(' - Processing frame ' + str(num_frame - initial_frame) + ' out of ' + str(last_frame - initial_frame))
            cv2.imwrite(dir_name + 'Originals/Original_Frame_' + str(num_frame) + '.jpg', original)
            cv2.imwrite(dir_name + 'Backgrounds/Background_Frame_' + str(num_frame) + '.jpg', back)
            cv2.imwrite(dir_name + 'Hulls/Hull_Frame_' + str(num_frame) + '.jpg', hull)

    num_frame = num_frame + 1

cap.release()