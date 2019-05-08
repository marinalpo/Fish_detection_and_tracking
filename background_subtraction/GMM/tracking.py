from fishutils import *
import matplotlib.image as mpimg
import cv2

ct = CentroidTracker()
(H, W) = (None, None)

video = ['Andratx8_6L', 'Andratx9_6L', 'CalaEgos5L']
num_video = 1

dir_name = '/Users/marinaalonsopoal/Documents/Telecos/Master/Research/Videos/'+video[num_video]+'/'

num_frame = 1
last_frame = 300

while num_frame <= last_frame:
    ori = mpimg.imread(dir_name + 'Originals/Original_Frame_' + str(num_frame) + '.jpg')
    hull = mpimg.imread(dir_name + 'Hulls/Hull_Frame_' + str(num_frame) + '.jpg')
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

    key = cv2.waitKey(10) & 0xFF
    num_frame += 1

cv2.destroyAllWindows()