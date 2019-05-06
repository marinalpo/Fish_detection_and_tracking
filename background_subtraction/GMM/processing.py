import matplotlib.image as mpimg
import cv2
import skimage.morphology as morph
from Display_Images import *

def processBackground(back):

    ret, binary = cv2.threshold(back, 127, 255, cv2.THRESH_BINARY)
    # Step 1: Reconstruction with Opening to remove little objects
    kernel = morph.disk(3)
    ope = morph.binary_opening(binary, kernel)
    rec = 255 * morph.reconstruction(ope, binary)
    rec = rec.astype('uint8')

    # Step 2: Closing to connect components
    kernel2 = morph.disk(20)
    clos = 255 * morph.binary_closing(rec, kernel2)
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

    return hull


num_frame = 1

while num_frame <= 301:
    back = mpimg.imread('/Users/marinaalonsopoal/Desktop/Backgrounds/Background_Frame_' + str(num_frame) + '.jpg')
    hull = processBackground(back)
    cv2.imwrite('/Users/marinaalonsopoal/Desktop/Hulls/Hull_Frame_' + str(num_frame) + '.jpg', hull)
    print('Printed frame ', num_frame)
    num_frame += 1