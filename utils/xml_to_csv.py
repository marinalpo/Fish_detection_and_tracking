import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

"""
    Code borrowed from: 
    https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py
"""
def xml_to_csv(path):
    xml_list = []
    print("EI")
    print(path)
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file) # this should be the first line
        root = tree.getroot()
        print(root)
        print("UEPA")
        for track in root.findall('track'):
            
            for box in track.findall('box'):
                print("Params")
                print(box.attrib)
                box_dict = box.attrib
                print("fish_type")
                print(box[0].text)
                fish_type = box[0].text
                # The format that the CSVDataloader expects is frame, x1, y1, x2, y2, class_name
                value = (int(box_dict['frame']), int(round(float(box_dict['xtl']))), int(round(float(box_dict['ytl']))), int(round(float(box_dict['xbr']))), int(round(float(box_dict['ybr']))), fish_type)
                xml_list.append(value)
    column_name = ['frame', 'xtl', 'ytl', 'xbr', 'ybr','fish_type']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    xml_path = os.path.join(os.getcwd(), '')
    xml_df = xml_to_csv(xml_path)
    xml_df.to_csv('raccoon_labels.csv', index=None)
    print('Successfully converted xml to csv.')

main()