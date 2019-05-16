"""
Functions to create, edit and write to a file a cvat compatible xml.
"""

import xml.etree.ElementTree as ET
import os.path

class cvatXml:
    """Xml object compatible with cvat"""
    def __init__(self, mode, labels, name, flipped='False', 
    width='', height=''):
        """
        Return an empty cvat xml.

        Arguments:
        mode: String (either 'annotation' or 'interpolation')
        labels: Dictionary of the labels as strings and their 
        attributes (as a list of strings)
        name: String, the name of the file
        
        Optional arguments:
        flipped : String, if the images were flipped, 'False' by default
        width: String, width of the images (for interpolation mode)
        height: String, height of the images (for interpolation mode)
        """
        root = ET.Element('annotations')
        xml = ET.ElementTree(root)
        meta = ET.Element('meta')
        root.append(meta)
        task = ET.Element('task')
        meta.append(task)
        modeElem = ET.Element('mode')
        modeElem.text = mode
        task.append(modeElem)
        nameElem = ET.Element('name')
        nameElem.text = name
        task.append(nameElem)
        labelsElem = ET.Element('labels')
        for label, attributes in labels.items():
            labelElem = ET.Element('label')
            nameElem = ET.Element('name')
            nameElem.text = label
            labelElem.append(nameElem)
            attributesElem = ET.Element('attributes') 
            for attribute in attributes:
                attributeElem = ET.Element('attribute')
                attributeElem.text = attribute
                attributesElem.append(attributeElem)
            labelElem.append(attributesElem)
            labelsElem.append(labelElem)
        task.append(labelsElem)

        if mode == "interpolation":
            original_size = ET.Element('original_size')
            widthElem = ET.Element('width')
            widthElem.text = width
            original_size.append(widthElem)
            heightElem = ET.Element('height')
            heightElem.text = height
            original_size.append(heightElem)
            task.append(original_size)

        self.mode = mode
        self.xml = xml

    def prettify(self, elem, level=0):
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
                self.prettify(elem, level+1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def addImage(self, imageId, width, height, name=''):
        """
        Add image element to an annotation mode xml

        Arguments:
        imageId: String, position of the image in the sequence (from 0)
        width: String, width of the image
        height: String, height of the image

        Optional arguments:
        name: String, name of the image
        """
        if self.mode == "annotation":
            image = ET.Element('image')
            image.set('id', imageId)
            image.set('name', name)
            image.set('width', width)
            image.set('height', height)
            self.xml.getroot().append(image)
        


    def addTrack(self, trackId, label):
        """
        Add a track to an interpolation mode xml

        Arguments:
        trackId: String, number of the track
        label: String, label for the track
        """
        if self.mode == "interpolation":
            track = ET.Element('track')
            track.set('id', trackId)
            track.set('label', label)
            self.xml.getroot().append(track)

    def addBoxToImage(self, imageId, label, xtl, ytl, xbr, ybr, occluded='0'):
        """
        Add a bounding box to an image

        Arguments:
        imageId: String, id of the image 
        label: String, label for the box
        xtl: String, x value for the 
        ytl: String, y value for top left corner
        xbr: String, x value for bottom right corner
        ybr: String, y value for bottom right corner

        Optional arguments:
        occluded: String, '1' or '0' (default '0')
        """
        for image in self.xml.getroot().findall('image'):
            if image.get('id') == imageId:
                box = ET.Element('box')
                box.set('label', label)
                box.set('xtl', xtl)
                box.set('ytl', ytl)
                box.set('xbr', xbr)
                box.set('ybr', ybr)
                box.set('occluded', occluded)
                image.append(box)
                break

    def addBoxToTrack(self, trackId, frame, xtl, ytl, xbr, ybr,
     outside='0', occluded='0', keyframe='1'):
        """
        Add a bounding box to a track

        Argument:
        trackId: String, id of the track
        frame: String, frame of the box
        xtl: String, x value for the 
        ytl: String, y value for top left corner
        xbr: String, x value for bottom right corner
        ybr: String, y value for bottom right corner

        Optional arguments:
        occluded: String, '1' or '0' (default '0')
        outside: String, '1' or '0' (default '0')
        keyframe: String, '1' or '0' (default '1')
        """
        for track in self.xml.getroot().findall('track'):
            if track.get('id') == trackId:
                box = ET.Element('box')
                box.set('frame', frame)
                box.set('xtl', xtl)
                box.set('ytl', ytl)
                box.set('xbr', xbr)
                box.set('ybr', ybr)
                box.set('occluded', occluded)
                box.set('outside', outside)
                box.set('keyframe', keyframe)
                track.append(box)
                break

    def writeXml(self, directory='', name=''):
        """
        Write the xml in a file

        Optional arguments:
        directory: String, path to the directory
        name: String, name of the file
        """
        self.prettify(self.xml.getroot())
        if name == '':
            name = self.xml.getroot().find('meta').find('task').find(
                'name').text+'.xml'
        path = os.path.join(directory, name)
        print("Writing the xml tree in the file %s..."%path, end = '')
        self.xml.write(path)
        print("[Done]")
    
    def getXml(self):
        """
        Access the xml
        """
        return self.xml

    def bboxInBounds(self): 
        """Correcting boxes partially or completely out of bounds,
        otherwise cvat will not be able to import the xml
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

        mode = self.xml.find('meta').find('task').find('mode').text          

        if mode == "interpolation":
            original_size = self.xml.getroot().find('meta').find('task').\
                            find('original_size')
            width = int(original_size.find('width').text)
            height = int(original_size.find('height').text)
            for track in self.xml.findall('track'):
                for bbox in track.findall('box'):
                    limitBbox(bbox)
                    
        if mode == "annotation":
            for image in self.xml.findall('image'):
                width = int(image.get("width"))
                height = int(image.get("height"))
                for bbox in image.findall('box'):
                    limitBbox(bbox)

