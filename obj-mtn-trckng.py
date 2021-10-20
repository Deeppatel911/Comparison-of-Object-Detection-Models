# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 18:57:22 2021

@author: DELL
"""

import cv2
import numpy as np

cap=cv2.VideoCapture('traffic.mp4')

#tracker=cv2.TrackerMOSSE_create() #opencv contrib python package-contains all tracker
# tracker=cv2.TrackerCSRT_create()
# success,img=cap.read()
# boundingBox=cv2.selectROI("Tracking",img,False)
# tracker.init(img,boundingBox)

success,frame1=cap.read()
success,frame2=cap.read()

# def drawBox(img,boundingBox):
    # x,y,w,h=int(boundingBox[0]),int(boundingBox[1]),int(boundingBox[2]),int(boundingBox[3])
    # cv2.rectangle(img,(x,y),((x+w),(y+h)),(255,0,255),3,1)
    # cv2.putText(img,"Tracking",(75,75),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)


while cap.isOpened():
    timer=cv2.getTickCount()
    success,img=cap.read()
    
    # success,boundingBox=tracker.update(img)
    
    # if success: 
       # drawBox(img,boundingBox)
    # else:
           # cv2.putText(img,"Lost",(75,75),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    
    diff=cv2.absdiff(frame1,frame2)
    gray=cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY) #it's easier to find contours in gray scale than bgr 
    blur=cv2.GaussianBlur(gray,(5,5),0) #2nd param-kernel size, 3rd param-sigma x value
    _,thresh=cv2.threshold(blur,20,255,cv2.THRESH_BINARY) #2nd param-threshold value,3rd param-max threshold value,4th param-type
    dilated=cv2.dilate(thresh,None,iterations=3) #2nd param-kernel
    contours,_=cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #2nd param-mode, 3rd param-method
    
    # cv2.drawContours(frame1,contours,-1,(0,255,0),2) #3rd param-contour id
    for c in contours:
        (x,y,w,h)=cv2.boundingRect(c)
        if cv2.contourArea(c)<350:
            continue
        cv2.rectangle(frame1,(x,y),((x+w),(y+h)),(0,255,0),3,1)
        cv2.putText(frame1,"Status: {}".format("Movement"),(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    
    fps=cv2.getTickFrequency()/(cv2.getTickCount()-timer)
    cv2.putText(img,str(int(fps)),(75,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    cv2.imshow("Tracking",img)
    
    cv2.imshow("Tracking",frame1)
    frame1=frame2
    ret,frame2=cap.read()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()  