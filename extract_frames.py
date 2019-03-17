import cv2 
import os

path = r"C:\Users\Usuario\Documents\Telecos\Master\Research\Videos"+"\\" # Video path
video_name = "Andratx9_6L"

# all_frames = 1 extracts all the frames from the video
# CAUTION! Normally the number of total frames is huge
all_frames = 0

n = 50 # Number of frames that you want to extract

if not os.path.exists(path + video_name):
	os.makedirs(path + video_name)

if all_frames == 1:
	vidcap = cv2.VideoCapture(path + video_name + '.MP4') 
	n = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

for x in range(1, n + 1):
	vidcap = cv2.VideoCapture(path + video_name + '.MP4') 
	success, image = vidcap.read()
	cv2.imwrite(os.path.join(path + video_name + "\\",video_name + '_frame%d.jpg' % x),image)

