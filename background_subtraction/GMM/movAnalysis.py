from fishutils import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

ct = CentroidTracker()
(H, W) = (None, None)

video = ['Andratx8_6L', 'Andratx9_6L', 'CalaEgos5L']
num_video = 0

dir_name = '/Users/marinaalonsopoal/Documents/Telecos/Master/Research/Videos/'+video[num_video]+'/'

num_frame = 200
last_frame = 300

fishEvo = {}  # Fish Evolution: Dictionary with KEY: Fish ID, VALUE: Concatenation tuple of [Frame][CenroidX, CentroidY}

print('Processing...')

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

fig, ax = plt.subplots()
for num_fish in range(len(fishEvo.keys())):
    evo = np.zeros([len(fishEvo[num_fish]), 2])
    for i in range(len(fishEvo[num_fish])):
        evo[i, :] = fishEvo[num_fish][i]
    isFish[num_fish] = classify(evo, 2)
    if isFish[num_fish]:
        ax.scatter(evo[:, 0], evo[:, 1], color='b')
    else:
        ax.scatter(evo[:, 0], evo[:, 1], color='r')
    ax.annotate(str(num_fish), (evo[0, 0], evo[0, 1]))
plt.title('Fish Location Evolution')
plt.show()

num_frame = 1
ct = CentroidTracker()
(H, W) = (None, None)

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
            print('id:', objectID, 'label: Fish', 'frame:', str(num_frame), 'xtl:', coordinates[1][0], 'ytl:',
              coordinates[1][1], 'xbr:', coordinates[1][0], 'ybr:', coordinates[1][0])
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