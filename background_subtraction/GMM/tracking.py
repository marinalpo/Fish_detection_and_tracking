from fishutils import *
import matplotlib.image as mpimg
import cv2

ct = CentroidTracker()
(H, W) = (None, None)

video = ['Andratx8_6L', 'Andratx9_6L', 'CalaEgos5L']
num_video = 2

dir_name = '/Users/marinaalonsopoal/Documents/Telecos/Master/Research/Videos/'+video[num_video]+'/'

num_frame = 1
last_frame = 150

while num_frame <= last_frame:

    ori = mpimg.imread(dir_name + 'Originals/Original_Frame_' + str(num_frame) + '.jpg')
    hull = mpimg.imread(dir_name + 'Hulls/Hull_Frame_' + str(num_frame) + '.jpg')

    # Get bounding boxes from detection
    rects = getBoxes(hull)

    objects = ct.update(rects)

    for (objectID, coordinates) in objects.items():
        print('id:', objectID, 'label: Fish', 'frame:', str(num_frame), 'xtl:', coordinates[1][0], 'ytl:',
              coordinates[1][1], 'xbr:', coordinates[1][2], 'ybr:', coordinates[1][3])
        text = "FISH {}".format(objectID)
        # ID Tag
        cv2.putText(ori, text, (coordinates[1][0], coordinates[1][1] - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
        # Bounding Box
        cv2.rectangle(ori, (coordinates[1][0], coordinates[1][1]), (coordinates[1][2], coordinates[1][3]), (255, 0, 0), 4)

    ori = cv2.cvtColor(ori, cv2.COLOR_BGR2RGB)
    cv2.imshow('Frame', ori)
    key = cv2.waitKey(10) & 0xFF
    num_frame += 1

cv2.destroyAllWindows()
