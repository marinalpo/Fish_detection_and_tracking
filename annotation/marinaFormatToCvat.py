"""
Convert marina format to a cvat compatible xml
(assumes 1080p images, change width and height
values in the code if necessary)


Usage: marinaFormatToCvat.py -i <pathToFile> -o <pathToXml> [-m <mode>]

Options:
-i=<pathToFile> --input  Path to the input file
-o=<pathToXml> --output  Path where the xml will be written
-m=<mode> --mode  Format of the xml (annotation or interpolation) 
                  [default: annotation]
"""
from cvatFile import cvatXml
from docopt import docopt
import os.path

width = '1920'
height = '1080'


arguments = docopt(__doc__)
with open(arguments['--input']) as file:
    labels = {}
    for line in file:
        labels[line.split(' ')[3]] = []
    name = os.path.basename(arguments['--output'].rsplit('.', 1)[0])
    xml = cvatXml(arguments['--mode'], labels, name, width=width, height=height)
    
    file.seek(0)

    if arguments['--mode'] == "annotation":
        previousFrame = -1
        for line in file:
            sline = line.split(' ')
            frame = int(sline[5])
            if frame != previousFrame:
                xml.addImage(sline[5], width, height)
            xml.addBoxToImage(sline[5], sline[3], sline[7], sline[9], sline[11], sline[13].rsplit('\n')[0])
            previousFrame = frame
    
    if arguments['--mode'] == "interpolation":
        tracks = 0
        for line in file:
            sline = line.split(' ')
            track = int(sline[1])
            if track >= tracks:
                xml.addTrack(sline[1], sline[3])
                tracks+=1
            xml.addBoxToTrack(sline[1], sline[5], sline[7], sline[9], sline[11], sline[13].rsplit('\n')[0])

    xml.bboxInBounds()
    xml.writeXml('', arguments['--output'])
