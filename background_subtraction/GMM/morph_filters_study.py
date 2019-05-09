import cv2
import skimage.morphology as morph
from Display_Images import *
import matplotlib.image as mpimg

video = ['Andratx8_6L', 'Andratx9_6L', 'CalaEgos5L']
num_video = 0
dir_name = '/Users/marinaalonsopoal/Documents/Telecos/Master/Research/Videos/'+video[num_video]+'/'
num_frame = 220

ori = mpimg.imread(dir_name + 'Originals/Original_Frame_' + str(num_frame) + '.jpg')
back = mpimg.imread(dir_name + 'Backgrounds/Background_Frame_' + str(num_frame) + '.jpg')

ret, binary = cv2.threshold(back, 127, 255, cv2.THRESH_BINARY)

# Step 1: Reconstruction with Opening to remove little objects
kernel = morph.disk(3)
ope1 = morph.binary_opening(binary, kernel)
out1 = 255 * morph.reconstruction(ope1, binary)
out1 = out1.astype('uint8')

# Step 2: Closing to connect components
kernel2 = morph.disk(1)
out2 = 255 * morph.binary_closing(out1, kernel2)
out2 = out2.astype('uint8')

# Step 3: Reconstruction with Opening to remove little objects
kernel3 = morph.disk(5)
ope2 = morph.binary_opening(out2, kernel3)
ope2 = 255 * ope2
ope2 = ope2.astype('uint8')
ret, ope2 = cv2.threshold(ope2, 127, 255, cv2.THRESH_BINARY)
out3 = 255 * morph.reconstruction(ope2, out2)
out3 = out3.astype('uint8')

# Step 4: Closing to connect components
kernel4 = morph.disk(10)
out4 = morph.binary_closing(out3, kernel4)

# Step 5: Minimum Hull computation
hull = 255 * morph.convex_hull_object(out4)
hull = hull.astype('uint8')


Display_6_Images(back, out1, out2, out3, out4, hull, 'Detected Foreground',
                 'Opening by Reconstruction', 'Closing', 'Opening by Reconstruction',
                 'Closing', 'Hulls', 'Video ' + str(video[num_video]) + ' Frame #' + str(num_frame))
