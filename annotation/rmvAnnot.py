"""

Remove all instances of a specific box after the indicated frame.
The box n is the n-th box to appear (indexed from 0).
Known problem: the box n is not always the n-th to appear...

Usage: rmvAnnot.py -f <pathtofile> -i <imputxml> [-o <outputxml>]
       rmvAnnot.py -i <inputxml> [-o <outputxml>] <box:frame>...
                   
Options:
 -i=<inputxml> --input  Path to the input xml file
 -o=<outputxml> --output  Path to the output xml file
 -f=<pathtofile> --file  Path to a <box:frame> sequences file

"""

from docopt import docopt
import xml.etree.ElementTree as ET
#Check that the script is called from the CLI
if __name__ == '__main__':
    #Load the CLI arguments in arguments
    arguments = docopt(__doc__)
    
    #Load the sequences
    sequences = {}
    if arguments['--file']:
        try:
            with open(arguments['--file'], 'r') as file:
                for line in file:
                    box, frame = line.split(':')
                    sequences[box] = frame
        except:
            raise Exception("Cannot read the <box:frame> sequences file %s"
                            %arguments['--file'])
    else:
        for sequence in arguments['<box:frame>']:
            box, frame = sequence.split(':')
            sequences[box] = frame


    #Load the xml
    xmlParse = ET.parse(arguments['--input'])

    #Find xml format (interpolation or annotation)
    mode = xmlParse.getroot().find('meta').find('task').find('mode').text
    
    ##Remove the specified boxes
    if mode == "interpolation":
        tracks = xmlParse.getroot().findall('track')
        for trackNumber, frame in sequences.items():
            track = tracks[int(trackNumber)]
            boxes = track.findall('box')
            boxesToThrow = [box for box in boxes
                            if int(box.get('frame'))>=int(frame)]
            for box in boxesToThrow:
                track.remove(box)

    if mode == "annotation":
        images = xmlParse.getroot().findall('image')
        for image in images:
            boxes = image.findall('box')
            boxesToThrow = []
            for boxNumber, frame in sequences.items():
                if int(image.get('id'))>=int(frame):
                    boxesToThrow.append(boxes[int(boxNumber)])
            for box in boxesToThrow:
                image.remove(box)

    #Write the result
    if arguments['--output']:
        print("Writing the xml tree in the file %s..."%
              arguments['--output'], end = '')
        xmlParse.write(arguments['--output'])
        print("[Done]")
    else:
        answer = input("This will overwrite the input xml. Continue ? (y/n)")
        if answer=='y':
            print("Writing the xml tree in the file %s..."%
                  arguments['--input'], end = '')
            xmlParse.write(arguments['--input'])
            print("[Done]")
        else:
            print("Program cancelled.")
        
                
    
