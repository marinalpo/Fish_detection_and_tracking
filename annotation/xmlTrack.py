"""

Guide a Re3 tracker through a sequence of images from a cvat xml
annotation file. Create a new annotation/interpolation file
including the result of the tracking.
Optional arguments can be alternatively configured in the python program.

Usage: vid2pics (-a | -t)[-i <images directory>][-x <xml file>]
                [-o <output xml>][-r <re3 directory>][-g] 
 
Options:
 -i=<images directory> --images  Path to the input video
 -x=<xml file> --xml  Path to an annotation xml
 -o=<output xml> --output  Path to the output xml file
 -a --annotation  Output in the cvat annotation format
 -t --interpolation  Output in the cvat interpolation format
 -r=<re3 directory> --re3dir  Path to the re3 directory
 -g --graphic  Show the tracking process

"""

import cv2
from docopt import docopt
import glob
import numpy as np
import os.path
import sys
import xml.etree.ElementTree as ET

#Paths to configure
basedir = os.path.abspath(os.path.dirname(__file__))
re3dir = "/home/utilisateur/builds/Re3/"
imageDir = "/home/utilisateur/Desktop/CalaEgos7LTest/slice1/"
imgFormat = ".jpg"
xmlFile = "7_xmlTrackTest.xml"
xmlPath = os.path.join(basedir, xmlFile)
newXmlFile = "trackedAnnotation.xml"
newXmlPath = os.path.join(basedir, newXmlFile)

def indent(elem, level=0):
    """
    Prettyprint xml formatter
    http://effbot.org/zone/element-lib.htm#prettyprint
    """
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

def bboxInBounds(newXml): 
    """Correcting boxes partially or completely out of bounds,
    otherwise cvat might not be able to import the xml
    """
    def limitBbox(bbox):
            xtags = ['xtl', 'xbr']
            ytags = ['ytl', 'ybr']
            for tag in xtags+ytags:
                if float(bbox.get(tag))<0:
                    bbox.set(tag, '0')
            for tag in xtags:
                if float(bbox.get(tag))>width:
                    bbox.set(tag, str(width))
            for tag in ytags:
                if float(bbox.get(tag))>height:
                    bbox.set(tag, str(height))
                    
    if arguments['--interpolation']:
        original_size = newXml.getroot().find('meta').find('task').\
                        find('original_size')
        width = int(original_size.find('width').text)
        height = int(original_size.find('height').text)
        for track in newXml.findall('track'):
            for bbox in track.findall('box'):
                limitBbox(bbox)
    if arguments['--annotation']:
        for image in newXml.findall('image'):
            width = int(image.get("width"))
            height = int(image.get("height"))
        for bbox in image.findall('box'):
            limitBbox(bbox)

def drawBboxes(bboxes, image):
    for bb,bbox in enumerate(bboxes):
        color = cv2.cvtColor(np.uint8([[[bb * 255 / len(bboxes), 128, 200]]]),
            cv2.COLOR_HSV2RGB).squeeze().tolist()
        cv2.rectangle(image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color, 2)
    cv2.imshow('Image', image)
    cv2.waitKey(1)

def getNewBboxes(tracker, ids, imageRGB, newIdsBboxes, newId):
    if len(ids)==1:
        #In case there is a single target to track
        #we use a different function
        if len(newIdsBboxes)>0:
            bboxes = tracker.track(ids[0], imageRGB, newIdsBboxes[newId])
            #In case there is a single bbox,
            #it needs to be explicitely included in a list
            bboxes = [bboxes]       
        else:
            bboxes = tracker.track(ids[0], imageRGB)
            bboxes = [bboxes]
    elif len(newIdsBboxes)>0:
        #Set the trackers
        #Start new tracks, but continue the others as well.
        #Only the new tracks need an initial bounding box.
        bboxes = tracker.multi_track(ids, imageRGB, newIdsBboxes)
    else:
        #All tracks are started, they don't need bounding boxes.
        bboxes = tracker.multi_track(ids, imageRGB)

    return bboxes

def addBboxesAnnotation(newXml, bboxes, ids, annotatedImgs,
                        annotationPos, ii, imagePaths):
    """Add bboxes in a cvat annotation format xml tree"""
    if annotationPos<len(annotatedImgs) and\
        ii+1 == int(annotatedImgs[annotationPos].get('id')):
        #Case where the next image is already annotated
        addBboxesToXml(newXml, bboxes, str(ii+1), ids)
    else:
        #Case where a new image element is created
        addBboxesToXml(newXml, bboxes, str(ii+1), ids,
        newImg=True, imageName=os.path.basename(imagePaths[ii]))

def addBboxesInterpolation(newXml, bboxes, ids, ii):
    """Add bboxes in a cvat interpolation format xml tree"""
    for bboxTags,bbox in zip(ids,bboxes):
        tags = bboxTags.split(':')
        idn = tags[2]
        trackNumber = idn
        #Compares the bbox id number to the number of tracks
        if  int(idn) >= len(newXml.findall('track')):
            #Create a new track element
            track = ET.Element("track")
            track.set("id", trackNumber)
            track.set("label", tags[0])
            newXml.getroot().append(track)
        box = ET.Element("box")
        box.set('frame', str(ii))
        box.set('outside', '0')
        box.set('occluded', tags[1])
        box.set('keyframe', '1')
        box.set('xtl', str(bbox[0]))
        box.set('ytl', str(bbox[1]))
        box.set('xbr', str(bbox[2]))
        box.set('ybr', str(bbox[3]))
        newXml.getroot().findall("track")[int(idn)].append(box)

def track(tracker, xmlParse, newXml, imagePaths):
    """
    Finds all bboxes in a cvat annotation file and tracks them
    """
    #Get all images with annotations
    annotatedImgs = xmlParse.findall('image')
    #Find the first annotated image
    try:
        firstAnnotated = int(annotatedImgs[0].get('id'))
    except:
        raise Exception("No annotated image in : %s"%xmlPath)

    print("Starting the tracking...")
    #Number of images to process
    numberImages = len(imagePaths)
    #The position in the annotated images list
    annotationPos = 0
    #The bounding box number, used for the ids
    idn=0
    #A list of bounding boxes ids
    ids = []
    #Start tracking on the first image where a fish has been annotated
    for ii,imagePath in enumerate(imagePaths[firstAnnotated:], firstAnnotated):
        image = cv2.imread(imagePath)
        # Tracker expects RGB, but opencv loads BGR.
        imageRGB = image[:,:,::-1]

        #Set the new bboxes to initiate trackers
        newIdsBboxes = {}
        if annotationPos<len(annotatedImgs):
            if ii == int(annotatedImgs[annotationPos].get('id')):
                annotatedImg = annotatedImgs[annotationPos]
                for box in annotatedImg.findall('box'):
                    newId = box.get('label')+':'+box.get('occluded')+\
                            ':'+str(idn)
                    idn+=1
                    x1 = float(box.get('xtl'))
                    y1 = float(box.get('ytl'))
                    x2 = float(box.get('xbr'))
                    y2 = float(box.get('ybr'))
                    newIdsBboxes[newId] = [x1, y1, x2, y2]
                    ids.append(newId)
                annotationPos+=1

        #Track the objects
        bboxes = getNewBboxes(tracker, ids, imageRGB, newIdsBboxes, newId)
            
        #Add the resulting bboxes to a xml
        if arguments['--annotation']:
            addBboxesAnnotation(newXml, bboxes, ids,
                                annotatedImgs,annotationPos, ii, imagePaths)
        if arguments['--interpolation']:
            addBboxesInterpolation(newXml, bboxes, ids, ii)
        if arguments['--graphic']:
            #Add the bboxes to the image (only useful to visualize)
            drawBboxes(bboxes, image)
        #Show the progress
        print("\rProcessing image %i/%i       "%(ii+1, numberImages), end='')
    #Create a new line for the prompt
    print()

def main(re3dir, imageDir, imgFormat, xmlPath, newXmlPath):
    #Import and create a re3_tracker object
    if arguments['--re3dir']:
        re3dir = arguments['--re3dir']
    sys.path.append(re3dir)
    from tracker import re3_tracker
    tracker = re3_tracker.Re3Tracker()

    #Load the images paths
    if arguments['--images']:
        imagePaths = arguments['--images']
    else:
        imagePaths = sorted(glob.glob(os.path.join(imageDir, '*'+imgFormat)))
    #Load the xml annotation file
    if arguments['--xml']:
        xmlPath = arguments['xml']
    print("Parsing the xml file %s..."%xmlPath, end='')
    xmlParse = ET.parse(xmlPath).getroot()
    print("[Done]")
    #Create a new XML object
    newXml = ET.parse(xmlPath)
    
    if arguments['--graphic']:
        #Window to visualize the tracking
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    
    #Remove images if we want an interpolation output
    if arguments['--interpolation']:
        meta = newXml.getroot().find('meta')
        original_size = ET.Element('original_size')
        width = ET.Element('width')
        width.text = xmlParse.find('image').get('width')
        height = ET.Element('height')
        height.text = xmlParse.find('image').get('height')
        original_size.append(width)
        original_size.append(height)
        task = meta.find('task')
        task.append(original_size)
        task.find('mode').text = 'interpolation'
        for image in newXml.getroot().findall('image'):
            newXml.getroot().remove(image)

    #Track the bboxes and store the result in newXml
    track(tracker, xmlParse, newXml, imagePaths)

    #Limit the bboxes to the images bounds
    bboxInBounds(newXml)
                
    #Prettify the xml
    indent(newXml.getroot())
    if arguments['--output']:
        newXmlPath = arguments['--output']
    #Write the xml tree in a file
    print("Writing the xml tree in the file %s..."%newXmlPath, end = '')
    newXml.write(newXmlPath)
    print("[Done]")

#checks that the script is called from the CLI
if __name__ == '__main__':
    #loads the CLI arguments in arguments
    arguments = docopt(__doc__, version='xmlTrack 0.2')
    main(re3dir, imageDir, imgFormat, xmlPath, newXmlPath)

    
