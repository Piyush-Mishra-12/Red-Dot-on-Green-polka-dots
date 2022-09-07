import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture("Video 1.mp4")

width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer= cv2.VideoWriter('Output1.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width,height))

while (True):
    ret, frame = cap.read()
    
    new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lg = np.array([35,65,0])#Lower values of Huge, Sauration and Value
    ug = np.array([60,255,255])#Higher values of Huge, Sauration and Value

    mask = cv2.inRange(new_frame,lg,ug)

    cordinates = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cordinates = imutils.grab_contours(cordinates)

    for c in cordinates:
        M = cv2.moments(c)
        if M["m00"] != 0:
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
        else:
            x, y = 0, 0
        
        result = cv2.circle(frame, (x,y), 10, (0, 0, 255), -1)
        
    cv2.imshow('result', result)
    writer.write(result)

    key = cv2.waitKey(30)
    if key == 27:
        break
writer.release()
cap.release()
cv2.destroyAllWindows()