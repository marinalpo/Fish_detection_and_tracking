import cv2
import skimage.morphology as morph
import numpy as np
from scipy.spatial import distance
from collections import OrderedDict
from pylab import *
import matplotlib.image as mpimg


#  Reference: https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
class CentroidTracker():
    def __init__(self, maxDisappeared = 25):
        self.nextObjectID = 0  # counter used to assign unique IDs to each object
        self.objects = OrderedDict()  # Dictionary where:
            # KEY: Object ID
            # VALUE: Tuple containing [Centroid (x,y),  Rectangle (startX, startY, endX, endY)]
        self.disappeared = OrderedDict()  # Object ID - #frames where object is lost
        self.maxDisappeared = maxDisappeared

    def register(self, coordinates):
        self.objects[self.nextObjectID] = coordinates
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):  # rects is a tuple formed by all the rectangles (startX, startY, endX, endY)
        if len(rects) == 0:
            for objectID in self.disappeared.keys():
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        # coordinates for the objects in the CURRENT frame
        inputCoordinates = {}
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCoordinates[i] = [(cX, cY), (startX, startY, endX, endY)]

        # if we are currently not tracking any object
        if len(self.objects) == 0:
            for i in range(0, len(inputCoordinates)):
                self.register(inputCoordinates[i])

        # Match the input centroids to existing object centroids
        else:
            objectIDs = list(self.objects.keys())

            objectCentroids = []  # already registered centroids
            for i in self.objects.keys():
                objectCentroids.append(self.objects[i][0])

            inputCentroids = []  # appearing centroids in the current frame
            for i in range(len(inputCoordinates)):
                inputCentroids.append(inputCoordinates[i][0])

            D = distance.cdist(objectCentroids, inputCentroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCoordinates[col]

                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCoordinates[col])
        return self.objects


def classify(evo, pix):
    isFish = False
    th = len(evo) * pix * 2
    diff = [np.diff(evo[:, 0]), np.diff(evo[:, 1])]
    mov = np.sum(np.abs(diff))  # absolute number of moved pixels
    if mov > th:
        isFish = True
    return isFish


def processBackground(back):

    ret, binary = cv2.threshold(back, 127, 255, cv2.THRESH_BINARY)

    # Step 1: Reconstruction with Opening to remove little objects
    kernel = morph.disk(2)
    ope1 = morph.binary_opening(binary, kernel)
    out1 = 255 * morph.reconstruction(ope1, binary)
    out1 = out1.astype('uint8')

    # Step 2: Closing to connect components
    kernel2 = morph.disk(1)
    out2 = 255 * morph.binary_closing(out1, kernel2)
    out2 = out2.astype('uint8')

    # Step 3: Reconstruction with Opening to remove little objects
    kernel3 = morph.disk(4)
    ope2 = morph.binary_opening(out2, kernel3)
    ope2 = 255 * ope2
    ope2 = ope2.astype('uint8')
    ret, ope2 = cv2.threshold(ope2, 127, 255, cv2.THRESH_BINARY)
    out3 = 255 * morph.reconstruction(ope2, out2)
    out3 = out3.astype('uint8')

    # Step 4: Closing to connect components
    kernel4 = morph.disk(20)
    out4 = morph.binary_closing(out3, kernel4)

    # Step 5: Minimum Hull computation
    hull = 255 * morph.convex_hull_object(out4)
    hull = hull.astype('uint8')
    return hull


def getBoxes(hull):
    ret, thresh = cv2.threshold(hull, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    rects = []
    for c in contours:
        box = cv2.boundingRect(c)
        startX = box[0]
        startY = box[1]
        endX = box[0] + box[2]
        endY = box[1] + box[3]
        rects.append([startX, startY, endX, endY])
    return rects



def plotCentroids(ori, hull, centroids, boxes, num_frame):
    font = cv2.FONT_HERSHEY_DUPLEX
    for i in range(len(boxes)):
        boxes = boxes.astype(np.int64)
        cv2.rectangle(ori, (boxes[i,0], boxes[i,1]), (boxes[i,2], boxes[i,3]), (0, 0, 255), 5)
        cv2.putText(ori, 'Id ' + str(i), (boxes[i, 0], boxes[i, 1]-10), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(ori, 'Andratx8_6L: Frame #' + str(num_frame), (10, 25), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Object Tracking', ori)
    cv2.waitKey(1)


def Display_Image(ima, name='Image'):
    f, axarr = plt.subplots(1, 1)
    # plt.suptitle(suptitle)
    axarr.imshow(ima, cmap='Greys_r')
    axarr.title.set_text(name)
    plt.show()


def Display_2_Images(ima1, ima2, name1='Image 1', name2='Image 2'):
    f, axarr = plt.subplots(2, 1)
    # plt.suptitle(suptitle)
    axarr[0].imshow(ima1, cmap='Greys_r')
    axarr[0].title.set_text(name1)
    axarr[1].imshow(ima2, cmap='Greys_r')
    axarr[1].title.set_text(name2)
    plt.show()


def Display_4_Images(ima1, ima2, ima3, ima4, name1='Image 1', name2='Image 2', name3='Image 3', name4='Image 4'):
    f, axarr = plt.subplots(2, 2)
    # plt.suptitle(suptitle)
    axarr[0, 0].imshow(ima1, cmap='Greys_r')
    axarr[0, 0].title.set_text(name1)
    axarr[0, 1].imshow(ima2, cmap='Greys_r')
    axarr[0, 1].title.set_text(name2)
    axarr[1, 0].imshow(ima3, cmap='Greys_r')
    axarr[1, 0].title.set_text(name3)
    axarr[1, 1].imshow(ima4, cmap='Greys_r')
    axarr[1, 1].title.set_text(name4)
    plt.show()


def Display_6_Images(ima1, ima2, ima3, ima4, ima5, ima6, name1='Image 1', name2='Image 2', name3='Image 3',
                     name4='Image 4', name5='Image 5', name6='Image 6', super=''):
    f, axarr = plt.subplots(2, 3)
    plt.suptitle(super)
    axarr[0, 0].imshow(ima1, cmap='Greys_r')
    axarr[0, 0].title.set_text(name1)
    axarr[0, 1].imshow(ima2, cmap='Greys_r')
    axarr[0, 1].title.set_text(name2)
    axarr[0, 2].imshow(ima3, cmap='Greys_r')
    axarr[0, 2].title.set_text(name3)
    axarr[1, 0].imshow(ima4, cmap='Greys_r')
    axarr[1, 0].title.set_text(name4)
    axarr[1, 1].imshow(ima5, cmap='Greys_r')
    axarr[1, 1].title.set_text(name5)
    axarr[1, 2].imshow(ima6, cmap='Greys_r')
    axarr[1, 2].title.set_text(name6)
    plt.show()

