import matplotlib.image as mpimg
import cv2
import numpy as np
from Display_Images import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def getCentroids(ima):
    ret, thresh = cv2.threshold(ima, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    centroids = np.zeros([len(contours), 2])
    boxes = np.zeros([len(contours), 4])  # x, y, w, h
    i = 0  # TODO: Delete i and make it with c
    for c in contours:
        M = cv2.moments(c)
        centroids[i, 0] = int(M["m10"] / M["m00"])
        centroids[i, 1] = int(M["m01"] / M["m00"])
        boxes[i, :] = cv2.boundingRect(c)
        boxes[i, 2] = boxes[i, 0] + boxes[i, 2]
        boxes[i, 3] = boxes[i, 1] + boxes[i, 3]
        # now startX, startY, endX, endY
        i = i + 1
    return centroids, boxes


def plotCentroids(ori, hull, centroids, boxes, num_frame):
    font = cv2.FONT_HERSHEY_DUPLEX
    for i in range(len(boxes)):
        boxes = boxes.astype(np.int64)
        cv2.rectangle(ori, (boxes[i,0], boxes[i,1]), (boxes[i,2], boxes[i,3]), (0, 0, 255), 5)
        cv2.putText(ori, 'Id ' + str(i), (boxes[i, 0], boxes[i, 1]-10), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(ori, 'Andratx8_6L: Frame #' + str(num_frame), (10, 25), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Object Tracking', ori)
    cv2.waitKey(1)


# num_frame = 1
#
# while num_frame <= 300:
#     ori = mpimg.imread('/Users/marinaalonsopoal/Desktop/Originals/Original_Frame_' + str(num_frame) + '.jpg', cv2.IMREAD_COLOR)
#     hull = mpimg.imread('/Users/marinaalonsopoal/Desktop/Hulls/Hull_Frame_' + str(num_frame) + '.jpg')
#     centroids, boxes = getCentroids(hull)
#     ori = cv2.cvtColor(ori, cv2.COLOR_BGR2RGB)
#     plotCentroids(ori, hull, centroids, boxes, num_frame)
#     num_frame = num_frame + 1
#
# cv2.destroyAllWindows()