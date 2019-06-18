import cv2
from fishutils import *
import matplotlib.image as mpimg
import argparse

# The videos that where used for this system where edited versions of the original ones, where the first minute of the
# video is removed to avoid camera settlement recordings, so the frame numbers do not match with the original ones.
video = ['Andratx8_6L', 'Andratx9_6L', 'CalaEgos5L']
num_video = 0  # Select number of video

# Directory from where the video will be extracted and where the processed frames will be saved
dir_name = '/Users/marinaalonsopoal/Documents/Telecos/Master/Research/Videos/'+video[num_video]+'/'
cap = cv2.VideoCapture(dir_name + video[num_video] + '.mp4')

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=200, nmixtures=5, backgroundRatio=0.7, noiseSigma=0)

capture = True

num_frame = 1
initial_frame = 150
last_frame = 155

print('Background Subtraction...')

while num_frame <= last_frame:
    ret, original = cap.read()

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
            print(' - Processing frame ' + str(num_frame-initial_frame) + ' out of ' + str(last_frame-initial_frame))
            cv2.imwrite(dir_name + 'Originals/Original_Frame_' + str(num_frame) + '.jpg', original)
            cv2.imwrite(dir_name + 'Backgrounds/Background_Frame_' + str(num_frame) + '.jpg', back)
            cv2.imwrite(dir_name + 'Hulls/Hull_Frame_' + str(num_frame) + '.jpg', hull)


    num_frame = num_frame + 1

cap.release()

ct = CentroidTracker()
(H, W) = (None, None)

num_frame = 1

fishEvo = {}  # Fish Evolution: Dictionary with KEY: Fish ID, VALUE: Concatenation tuple of [Frame][CenroidX, CentroidY}

print('Fish Tracking...')

# Initial Tracking (obtain position and id info to discard later)
while num_frame <= last_frame:
    hull = mpimg.imread(dir_name + 'Hulls/Hull_Frame_' + str(num_frame) + '.jpg')
    # Get bounding boxes from detection
    rects = getBoxes(hull)
    objects = ct.update(rects)
    for (objectID, coordinates) in objects.items():
        newInfo = [coordinates[1][0], coordinates[1][1]]
        if objectID in fishEvo:
            fishEvo[objectID] += [newInfo, ]
        else:
            fishEvo[objectID] = [newInfo]
    num_frame += 1

isFish = {}

for num_fish in range(len(fishEvo.keys())):
    evo = np.zeros([len(fishEvo[num_fish]), 2])
    for i in range(len(fishEvo[num_fish])):
        evo[i, :] = fishEvo[num_fish][i]
    isFish[num_fish] = classify(evo, 2)

num_frame = 1
ct = CentroidTracker()
(H, W) = (None, None)

print('Fish Classification...')
while num_frame <= last_frame:
    ori = mpimg.imread(dir_name + 'Originals/Original_Frame_' + str(num_frame) + '.jpg')
    hull = mpimg.imread(dir_name + 'Hulls/Hull_Frame_' + str(num_frame) + '.jpg')
    # Get bounding boxes from detection
    rects = getBoxes(hull)
    objects = ct.update(rects)
    for (objectID, coordinates) in objects.items():
        text = "ID {}".format(objectID)
        color = (255, 0, 0)
        if isFish[objectID]:
            #print('id:', objectID, 'label: Fish', 'frame:', str(num_frame), 'xtl:', coordinates[1][0], 'ytl:',
            #  coordinates[1][1], 'xbr:', coordinates[1][0], 'ybr:', coordinates[1][0])




            # RETINANET!!!



            color = (0, 0, 255)
            # ID Tag
            cv2.putText(ori, text, (coordinates[1][0], coordinates[1][1] - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            # Bounding Box
            cv2.rectangle(ori, (coordinates[1][0], coordinates[1][1]), (coordinates[1][2], coordinates[1][3]),
                          color, 4)

    ori = cv2.cvtColor(ori, cv2.COLOR_BGR2RGB)
    cv2.imshow('Fish Detection', ori)
    key = cv2.waitKey(10) & 0xFF
    num_frame += 1

cv2.destroyAllWindows()