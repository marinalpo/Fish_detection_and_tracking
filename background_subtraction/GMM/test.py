from Tracking import *
from centroid_tracking import *
import matplotlib.image as mpimg
import numpy as np
from imutils.video import VideoStream
import imutils
import cv2
import time
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0), (0, 255, 255), (255, 140, 0), (0, 128, 0),
          (138, 43, 226), (210, 205, 30), (255, 192, 203), (259, 69, 0), (0, 0, 0), (255, 255, 255), (0, 0, 255), (255, 0, 0),
          (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)]

ct = CentroidTracker()
(H, W) = (None, None)

num_frame = 75
while num_frame <= 250:
    ori = mpimg.imread('/Users/marinaalonsopoal/Desktop/Originals/Original_Frame_' + str(num_frame) + '.jpg', cv2.IMREAD_COLOR)
    hull = mpimg.imread('/Users/marinaalonsopoal/Desktop/Hulls/Hull_Frame_' + str(num_frame) + '.jpg')
    centroids, boxes = getCentroids(hull)
    rects = []

    for i in range(0, len(centroids)):
        box = boxes[i,:]
        rects.append(box)
        (startX, startY, endX, endY) = box.astype("int")
        cv2.rectangle(ori, (startX, startY), (endX, endY), (255,0,0), 4)

    objects = ct.update(rects)


    for (objectID, centroid) in objects.items():
        text = "FISH {}".format(objectID)
        cv2.putText(ori, text, (centroid[0]-10, centroid[1]-10),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
        cv2.circle(ori, (centroid[0], centroid[1]), 4,  (255, 0, 0), -1)

    cv2.putText(ori, 'Andratx8_6L: Frame #' + str(num_frame), (10, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    ori = cv2.cvtColor(ori, cv2.COLOR_BGR2RGB)
    cv2.imshow("Frame", ori)

    key = cv2.waitKey(1) & 0xFF
    num_frame += 1

cv2.destroyAllWindows()