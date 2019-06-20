import cv2
import numpy as np


img = cv2.imread('/imatge/ppalau/work/Fishes/joint_data/Original_Frame_994.jpg')
bbox =[1211, 174, 1228, 195]
cropped_img = img[bbox[1]-50:bbox[3]+50, bbox[0]-100:bbox[2]+100,:]
cv2.imwrite('cropped_image.jpg', cropped_img)

