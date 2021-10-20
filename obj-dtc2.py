# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 18:39:46 2021

@author: DELL
"""

import cv2
import numpy as np

def nothing():
    pass

def getContours(img):
    contours, heirarchy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area=cv2.contourArea(cnt)
        #print(area)
        if area>50:
            cv2.drawContours(imgContour,cnt,-1,(255,0,0),3)
            peri=cv2.arcLength(cnt,True)
            approx=cv2.approxPolyDP(cnt,0.02*peri,True)
            #print(len(approx))
            objCor=len(approx)
            x,y,w,h=cv2.boundingRect(approx)
            
            if objCor==3:
                objectType="triangle"
            elif objCor==4:
                aspRatio=w/float(h)
                if aspRatio>0.95 and aspRatio<1.05:
                    objectType="Square"
                else:
                    objectType="Rectangle"
            elif objCor==5:
                objectType="Pentagon"
            elif objCor==6:
                objectType="Hexagon"
            elif objCor==7:
                objectType="Heptagon"
            elif objCor==8:
                objectType="Octagon"
            elif objCor>8:
                objectType="Circle"
            else:
                objectType="None"
            
            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(imgContour,objectType,(x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)
            
            
#cap=cv2.VideoCapture(0)

while True:
    path="shapes.png"
    frame=cv2.resize(cv2.imread(path),(640,480))
    imgContour=frame.copy()
    #success,frame=cap.read()
    
    imgGray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    imgBlur=cv2.GaussianBlur(imgGray,(7,7),1)
    imgCanny=cv2.Canny(frame,50,50)
    
    
    getContours(imgCanny)
    
    cv2.imshow("Video",frame)
    cv2.imshow("gray",imgGray)
    cv2.imshow("blur",imgBlur)
    cv2.imshow("canny",imgCanny)
    cv2.imshow("contour",imgContour)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cap.release()
cv2.destroyAllWindows()    