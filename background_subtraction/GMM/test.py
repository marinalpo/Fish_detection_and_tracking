import matplotlib.image as mpimg
import cv2
import skimage.morphology as morph
from Display_Images import *

num_frame1 = 50
num_frame2 = 55
num_frame3 = 60

original1 = mpimg.imread('/Users/marinaalonsopoal/Desktop/Original_Frame_' + str(num_frame1) + '.jpg')
original2 = mpimg.imread('/Users/marinaalonsopoal/Desktop/Original_Frame_' + str(num_frame2) + '.jpg')
original3 = mpimg.imread('/Users/marinaalonsopoal/Desktop/Original_Frame_' + str(num_frame3) + '.jpg')
hull1 = mpimg.imread('/Users/marinaalonsopoal/Desktop/Hull_Frame_' + str(num_frame1) + '.jpg')
hull2 = mpimg.imread('/Users/marinaalonsopoal/Desktop/Hull_Frame_' + str(num_frame2) + '.jpg')
hull3 = mpimg.imread('/Users/marinaalonsopoal/Desktop/Hull_Frame_' + str(num_frame3) + '.jpg')

Display_6_Images(original1, original2, original3, hull1, hull2, hull3, 'ori1', 'ori2', 'ori3', 'hull1', 'hull2', 'hull3')