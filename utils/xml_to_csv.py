import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

"""
    Code borrowed from: 
    https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py
"""
base_image_path = "/work/morros/fish/video_segments"

def xml_to_csv(path):
    xml_list = []
    print("EI")
    print(path)
    for xml_file in glob.glob(path + '/*.xml'):
        
        filename = xml_file.split('\\')
        print(filename)
        complete_videoname = filename[-1].split('_')
        videoname = complete_videoname[0]
        videopart = complete_videoname[1].split('.')[0]
        print("Filename")
        print(xml_file)
        print("Video")
        print(videoname)
        print("Video part")
        print(videopart)
              
        tree = ET.parse(xml_file) # this should be the first line
        root = tree.getroot()
        print(root)
        for track in root.findall('image'):
            print(track.attrib)
            image_dict = track.attrib
            image_name = image_dict['name']
            complete_image_name = os.path.join(base_image_path, videoname,image_name)

            for box in track.findall('box'):
                box_dict = box.attrib
                print("box_dict")
                print(box_dict)
                # The format that the CSVDataloader expects is frame, x1, y1, x2, y2, class_name
                value = (str(complete_image_name), int(round(float(box_dict['xtl']))), int(round(float(box_dict['ytl']))), int(round(float(box_dict['xbr']))), int(round(float(box_dict['ybr']))), str(box_dict['label']))
                xml_list.append(value)
    column_name = ['frame', 'xtl', 'ytl', 'xbr', 'ybr','fish_type']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    xml_path = os.path.join(os.getcwd(), '')
    print(xml_path)
    xml_df = xml_to_csv(xml_path)
    xml_df.to_csv('raccoon_labels.csv', index=None)
    print('Successfully converted xml to csv.')

main()