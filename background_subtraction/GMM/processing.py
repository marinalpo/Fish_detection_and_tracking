import matplotlib.image as mpimg
import cv2
import skimage.morphology as morph
import numpy as np
from Display_Images import *

num_frame = 145
original = mpimg.imread('/Users/marinaalonsopoal/Desktop/Original_Frame_' + str(num_frame) + '.jpg')
back = mpimg.imread('/Users/marinaalonsopoal/Desktop/Back_Frame_' + str(num_frame) + '.jpg')
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
mark[size[0]-1, :] = 0
mark[:, size[1]-1] = 0
mark = 255-mark
rec2 = 255 - rec
durec = morph.reconstruction(mark, rec2)
durec = 255 - durec
# TODO Fix case when fish is in the background (apply some mask)

# Step 3: Simple Closing to perfectionate the filling
kernel = morph.disk(10)
clos = morph.binary_closing(durec, kernel)

# Display_4_Images(back, rec, durec, clos, 'Original', 'Reconstruction with Opening', 'Dual Reconstruction with Frame', 'Closing')

hull = morph.convex_hull_object(clos)
# cv2.imwrite('/Users/marinaalonsopoal/Desktop/Hull_Frame_' + str(num_frame) + '.jpg', clos)
Display_2_Images(back, hull, 'Processed Image', 'Object Polygon Hull')