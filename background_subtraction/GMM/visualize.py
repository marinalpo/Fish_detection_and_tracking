import cv2
from processing import *
import matplotlib.image as mpimg

video = ['Andratx8_6L', 'Andratx9_6L', 'CalaEgos5L']
num_video = 0

dir_name = '/Users/marinaalonsopoal/Documents/Telecos/Master/Research/Videos/'+video[num_video]+'/'

num_frame = 1
last_frame = 1000

while num_frame <= last_frame:

    hull = mpimg.imread(dir_name + 'Hulls/Hull_Frame_' + str(num_frame) + '.jpg')
    cv2.putText(hull, 'Frame #' + str(num_frame), (15, 80),
                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
    k = cv2.waitKey(10) & 0xff
    cv2.imshow('Object Hulls', hull)
    if k == 27:
        break

    num_frame = num_frame + 1

