import matplotlib.image as mpimg
import cv2
import skimage.morphology as morph
import numpy as np
from Display_Images import *


def process_back(back):
    ret, binary = cv2.threshold(back, 127, 255, cv2.THRESH_BINARY)
    # Step 1: Reconstruction with Opening
    kernel = morph.disk(8)
    ope = morph.binary_opening(binary, kernel)
    rec = morph.reconstruction(ope, binary)
    # Step 2: Dual Reconstruction with Frame Marker
    mark = 0 * np.copy(back) + 255
    size = back.shape
    mark[1, :] = 0
    mark[:, 1] = 0
    mark[size[0] - 1, :] = 0
    mark[:, size[1] - 1] = 0
    mark = 255 - mark
    rec2 = 255 - rec
    durec = morph.reconstruction(mark, rec2)
    durec = 255 - durec
    # Step 3: Simple Closing to perfectionate the filling
    kernel = morph.disk(10)
    clos = morph.binary_closing(durec, kernel)
    hull = morph.convex_hull_object(clos)
    return hull

num_frame1 = 145
num_frame2 = 160

back1 = mpimg.imread('/Users/marinaalonsopoal/Desktop/Back_Frame_' + str(num_frame1) + '.jpg')
back2 = mpimg.imread('/Users/marinaalonsopoal/Desktop/Back_Frame_' + str(num_frame2) + '.jpg')

hull1 = process_back(back1)
hull2 = process_back(back2)

Display_4_Images(back1, hull1, back2, hull2, 'Background Frame #' + str(num_frame1), 'Polygons Frame #'+ str(num_frame1), 'Background Frame #' + str(num_frame2), 'Polygons Frame #' + str(num_frame2))