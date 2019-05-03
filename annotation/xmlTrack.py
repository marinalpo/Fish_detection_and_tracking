"""

Guide a Re3 tracker through a sequence of images from a cvat xml
annotation file. Create a new annotation file including the
result of the tracking. 

"""

import cv2
import glob
import numpy as np
import os.path
import xml.etree.ElementTree as ET

#Absolute path to the script directory (in Re3 build directory...)
basedir = os.path.abspath(os.path.dirname(__file__))
#Import re3_tracker functions from the tracker directory
from tracker import re3_tracker

tracker = re3_tracker.Re3Tracker()

# in-place prettyprint formatter
def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def addBboxesToXml(elementTree, bboxes,
                   imageId, ids, newImg=False, imageName=''):
    """Add bboxes in a xml tree. Add a new image if required"""
    #Position of the image in the file
    #From the first annotated image on, every frame
    #will have added annotation
    pos = int(imageId)-int(elementTree.getroot().find("image").get("id"))
    #Create a new image element
    if newImg:
        image = ET.Element("image")
        image.set("id", imageId)
        image.set("name", imageName)
        previousImg = elementTree.getroot().findall("image")[pos-1]
        image.set("width", previousImg.get("width"))
        image.set("height", previousImg.get("height"))
        #+2 is because of <version> and <meta> before the images
        elementTree.getroot().insert(pos+2, image)
    for bboxTags,bbox in zip(ids,bboxes):
        box = ET.Element("box")
        tags = bboxTags.split(':')
        box.set("label", tags[0])
        box.set("occluded", tags[1])
        box.set('xtl', str(bbox[0]))
        box.set('ytl', str(bbox[1]))
        box.set('xbr', str(bbox[2]))
        box.set('ybr', str(bbox[3]))
        elementTree.getroot().findall("image")[pos].append(box)

#Load the images paths 
imageDir = "/home/utilisateur/Desktop/share/CalaEgos8L/0010/"
imagePaths = sorted(glob.glob(os.path.join(imageDir, '*.jpg')))
#Load the xml annotation file
xmlpath = os.path.join(basedir, "6_CalaEgos8L_0010.xml")
xmlparse = ET.parse(xmlpath).getroot()

#Find the first annotated image
#Can we not just take the first image ? TO CHECK
annotatedImgs = xmlparse.findall('image')
try:
    firstAnnotated = int(annotatedImgs[0].get('name').split('.')[0])
    imagePos = 0
except:
    raise Exception("No image found in the xml file !! "
                    "(or their names are not numbers)")
for i, image in enumerate(annotatedImgs):
    imageNumber = int(image.get('name').split('.')[0])
    if imageNumber < firstAnnotated:
        firstAnnotated = imageNumber
        imagePos = i
#Find the position of the first annotated image in the list imagePaths
#Can we not just use the id of the first annotated image ? TO CHECK        
try:
    firstAnnotationPos = firstAnnotated-\
                         int(os.path.splitext(
                             os.path.basename(imagePaths[0]))[0])
except:
    raise Exception("no image named <number>.jpg found in %s"%imageDir)

#Window to visualize the tracking
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

#Create a new XML object
newXml = ET.parse(xmlpath)

#The position in the annotated images list
annotationPos = firstAnnotationPos
#The bounding box number, used for the ids
idn=0
#A list of bounding boxes ids
ids = []
#Start tracking on the first image where a fish has been annotated
for ii,imagePath in enumerate(imagePaths[firstAnnotationPos:]):
    image = cv2.imread(imagePath)
    # Tracker expects RGB, but opencv loads BGR.
    imageRGB = image[:,:,::-1]

    #Set the new bboxes to initiate trackers
    newIdsBboxes = {}
    if annotationPos<len(annotatedImgs):
        if ii == int(annotatedImgs[annotationPos].get('id')):
            annotatedImg = annotatedImgs[annotationPos]
            for box in annotatedImg.findall('box'):
                newId = box.get('label')+':'+box.get('occluded')+':'+str(idn)
                idn+=1
                if newId not in ids:
                    x1 = float(box.get('xtl'))
                    y1 = float(box.get('ytl'))
                    x2 = float(box.get('xbr'))
                    y2 = float(box.get('ybr'))
                    newIdsBboxes[newId] = [x1, y1, x2, y2]
                    ids.append(newId)
            annotationPos+=1

    if len(newIdsBboxes)>0:
        print("new boxes !")
        print(newIdsBboxes)
        #Set the trackers
        #Start new tracks, but continue the others as well.
        #Only the new tracks need an initial bounding box.
        bboxes = tracker.multi_track(ids, imageRGB, newIdsBboxes)
        #In this case annotationPos has already been incremented
        nextPos = annotationPos
    else:
        #All tracks are started, they don't need bounding boxes.
        bboxes = tracker.multi_track(ids, imageRGB)
        nextPos = annotationPos+1
        
    #Add the resulting bboxes to a xml
    if nextPos<len(annotatedImgs) and\
        ii+1 == int(annotatedImgs[nextPos].get('id')):
        #Case where the next image is already annotated
        addBboxesToXml(newXml, bboxes, str(ii+1), ids)
    else:
        #Case where a new image element is created
        addBboxesToXml(newXml, bboxes, str(ii+1), ids,
        newImg=True, imageName=str(int(os.path.splitext(
            os.path.basename(imagePaths[0]))[0])))

    #Add the bboxes to the image (only useful to visualize)
    for bb,bbox in enumerate(bboxes):
        color = cv2.cvtColor(np.uint8([[[bb * 255 / len(bboxes), 128, 200]]]),
            cv2.COLOR_HSV2RGB).squeeze().tolist()
        cv2.rectangle(image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color, 2)
    cv2.imshow('Image', image)
    cv2.waitKey(1)

indent(newXml.getroot())
newXml.write(os.path.join(basedir, "trackedAnnotation.xml"))
    
