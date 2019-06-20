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
    xml_list_train = []
    xml_list_val = []
    serranus = [0, 0] # Number of serranus in validation and training set
    other = [0, 0] # Number of ohter fishes in validation and training set
    for i, xml_file in enumerate(glob.glob(path + '/*.xml')):
        if(i==0 or i==3):
            # Now do the validation set
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
                # print(track.attrib)
                image_dict = track.attrib
                image_name = image_dict['name']
                complete_image_name = os.path.join(base_image_path, videoname,videopart,image_name)
                complete_image_name = complete_image_name.replace(os.sep, '/')
                for box in track.findall('box'):
                    box_dict = box.attrib
                    # print("box_dict")
                    # print(box_dict)
                    if(box_dict['label'] == 'serranus'):
                        serranus[0] += 1
                    else: 
                        other[0] += 1
                    # The format that the CSVDataloader expects is frame, x1, y1, x2, y2, class_name
                    value = (str(complete_image_name), int(round(float(box_dict['xtl']))), int(round(float(box_dict['ytl']))), int(round(float(box_dict['xbr']))), int(round(float(box_dict['ybr']))), str(box_dict['label']))
                    xml_list_val.append(value)
        else:
            # Now do the training set
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
            # print(root)
            for track in root.findall('image'):
                # print(track.attrib)
                image_dict = track.attrib
                image_name = image_dict['name']
                complete_image_name = os.path.join(base_image_path, videoname,videopart,image_name)
                complete_image_name = complete_image_name.replace(os.sep, '/')
                for box in track.findall('box'):
                    box_dict = box.attrib
                    # print("box_dict")
                    # print(box_dict)
                    if(box_dict['label'] == 'serranus'):
                        serranus[1] += 1
                    else: 
                        other[1] += 1
                    # The format that the CSVDataloader expects is frame, x1, y1, x2, y2, class_name
                    value = (str(complete_image_name), int(round(float(box_dict['xtl']))), int(round(float(box_dict['ytl']))), int(round(float(box_dict['xbr']))), int(round(float(box_dict['ybr']))), str(box_dict['label']))
                    xml_list_train.append(value)

    column_name = ['frame', 'xtl', 'ytl', 'xbr', 'ybr','fish_type']
    xml_df_val = pd.DataFrame(xml_list_val, columns=None)
    xml_df_train = pd.DataFrame(xml_list_train, columns = None)
    print("Percentage of serranus in training set = "+ str(serranus[1]/(serranus[1]+other[1])))
    print("Number of serranus in training = " + str(serranus[1]))
    print("Percentage of serranus in validation set = "+ str(serranus[0]/(serranus[0]+other[0])))
    print("Number of serranus in validation = " + str(serranus[0]))
    return xml_df_val, xml_df_train


def main():
    xml_path = os.path.join(os.getcwd(), '')
    print(xml_path)
    xml_df_val, xml_df_train = xml_to_csv(xml_path)
    xml_df_val.to_csv('val.csv', index=None)
    xml_df_train.to_csv('train.csv', index=None)
    print('Successfully converted xml to csv.')

main()