import cv2
import skimage.morphology as morph
from Display_Images import *
import matplotlib.image as mpimg

video = ['Andratx8_6L', 'Andratx9_6L', 'CalaEgos5L']
num_video = 0
dir_name = '/Users/marinaalonsopoal/Documents/Telecos/Master/Research/Videos/'+video[num_video]+'/'
num_frame = 250

ori = mpimg.imread(dir_name + 'Originals/Original_Frame_' + str(num_frame) + '.jpg')
back = mpimg.imread(dir_name + 'Backgrounds/Background_Frame_' + str(num_frame) + '.jpg')

ret, binary = cv2.threshold(back, 127, 255, cv2.THRESH_BINARY)

# # Step 1: Reconstruction with Opening to remove little objects
# kernel = morph.disk(3)
# ope = morph.binary_opening(binary, kernel)
# rec = 255 * morph.reconstruction(ope, binary)
# rec = rec.astype('uint8')

# Step 2: Closing to connect components
kernel2 = morph.disk(5)
clos = 255 * morph.binary_closing(back, kernel2)
clos = clos.astype('uint8')

# Step 3: Reconstruction with Opening to remove little objects
kernel3 = morph.disk(5)
ope2 = morph.binary_opening(clos, kernel3)
ope2 = 255 * ope2
ope2 = ope2.astype('uint8')
ret, ope2 = cv2.threshold(ope2, 127, 255, cv2.THRESH_BINARY)
rec2 = 255 * morph.reconstruction(ope2, clos)
rec2 = rec2.astype('uint8')

# Step 4: Minimum Hull computation
hull = 255 * morph.convex_hull_object(rec2)
hull = hull.astype('uint8')

Display_6_Images(ori, back, back, clos, rec2, hull)