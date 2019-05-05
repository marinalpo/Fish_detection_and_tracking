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
        i = i + 1
    return centroids, boxes


num_frame1 = 145
ori1 = mpimg.imread('/Users/marinaalonsopoal/Desktop/Original_Frame_' + str(num_frame1) + '.jpg')
hull1 = mpimg.imread('/Users/marinaalonsopoal/Desktop/Hull_Frame_' + str(num_frame1) + '.jpg')

centroids1, boxes1 = getCentroids(hull1)


f, axarr = plt.subplots(2, 1)
axarr[0].imshow(ori1, cmap='Greys_r')
axarr[0].title.set_text('Original Frame #145')
for i in range(len(boxes1)):
    rect = patches.Rectangle((boxes1[i,0], boxes1[i,1]),boxes1[i,2],boxes1[i,3],linewidth=2,edgecolor='r',facecolor='none')
    axarr[0].add_patch(rect)

axarr[1].imshow(hull1, cmap='Greys_r')
for i in range(len(boxes1)):
    axarr[1].scatter(centroids1[i,0], centroids1[i,1], color='r')
axarr[1].title.set_text('Centroids Frame #145')
plt.show()