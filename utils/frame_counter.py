#!/usr/bin/python
# -*- coding: utf-8 -*
import numpy as np
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(r"C:/python27/CalaEgos6L_short.mp4")
counter = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here

    # Display the resulting frame
    if(counter == 4304):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame_rgb)
        plt.show()
    counter += 1    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()