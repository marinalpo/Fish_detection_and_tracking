import cv2

cap = cv2.VideoCapture('/Users/marinaalonsopoal/Documents/Telecos/Master/Research/Videos/Original/Andratx8_6L.MP4')

img_array = []
num_frame = 100

while num_frame <= 250:
    img = cv2.imread('/Users/marinaalonsopoal/Desktop/Backgrounds/Hull_Frame_'+str(num_frame)+'.jpg')
    img = cv2.imread('/Users/marinaalonsopoal/Desktop/Tracking/Tracking_Frame_' + str(num_frame) + '.pdf')
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)
    num_frame = num_frame+1

out = cv2.VideoWriter('tracking.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
# Go to https://video.online-convert.com/convert-to-mp4 to convert to MP4

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()