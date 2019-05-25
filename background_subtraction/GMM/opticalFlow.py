import cv2
import numpy as np
video = ['Andratx8_6L', 'Andratx9_6L', 'CalaEgos5L']
num_video = 0

dir_name = '/Users/marinaalonsopoal/Documents/Telecos/Master/Research/Videos/'+video[num_video]+'/'
cap = cv2.VideoCapture(dir_name + video[num_video] + '.mp4')
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

while(1):
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    # bw = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    # ret, thresh1 = cv2.threshold(bw, 75, 255, cv2.THRESH_BINARY)

    cv2.imshow('frame2', hsv)
    cv2.imshow('Ori', frame2)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png', frame2)
        cv2.imwrite('opticalhsv.png', rgb)
    prvs = next

cap.release()
cv2.destroyAllWindows()