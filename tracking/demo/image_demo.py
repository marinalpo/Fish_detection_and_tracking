import cv2
import glob
import numpy as np
import sys
import os.path

basedir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
from tracker import re3_resnet_tracker2 as re3_tracker
#re3_resnet_tracker as

#if not os.path.exists(os.path.join(basedir, 'data')):
#    import tarfile
#    tar = tarfile.open(os.path.join(basedir, 'data.tar.gz'))
#    tar.extractall(path=basedir)

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('Image', 640, 480)
tracker = re3_tracker.Re3Tracker()
#image_paths = sorted(glob.glob(os.path.join(
#    os.path.dirname(__file__), 'data', '*.jpg')))
#
image_paths = sorted(glob.glob(os.path.join(
    os.path.dirname(__file__), 'FishTest', 'imgTest', '*.jpg')))
#
#initial_bbox = [175, 154, 251, 229]
#initial_bbox =[1644, 146, 1644+275, 146+159]
initial_bbox=[434.55, 232.30, 482.40, 274.20]

tracker.track('ball', image_paths[0], initial_bbox)
for image_path in image_paths:
    image = cv2.imread(image_path)
    # Tracker expects RGB, but opencv loads BGR.
    imageRGB = image[:,:,::-1]
    bbox = tracker.track('ball', imageRGB)
    print(initial_bbox)
    print(bbox)
    cv2.rectangle(image,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            [0,0,255], 2)
    cv2.imshow('Image', image)
    cv2.waitKey(1)