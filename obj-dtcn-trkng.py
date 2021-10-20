# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 16:42:15 2021

@author: DELL
"""

import cv2
import numpy as np

def nothing():
    pass

#cap=cv2.VideoCapture(0)

cv2.namedWindow("tracking")
cv2.createTrackbar("min hue","tracking",0,255,nothing)
cv2.createTrackbar("min sat","tracking",0,255,nothing)
cv2.createTrackbar("min value","tracking",0,255,nothing)
cv2.createTrackbar("max hue","tracking",255,255,nothing)
cv2.createTrackbar("max sat","tracking",255,255,nothing)
cv2.createTrackbar("max value","tracking",255,255,nothing)

while True:
    frame=cv2.resize(cv2.imread("w motors lykan hypersport.jpg"),(640,480))
    #success,frame=cap.read()
    
    img_hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    lh=cv2.getTrackbarPos("min hue","tracking")
    ls=cv2.getTrackbarPos("min sat","tracking")
    lv=cv2.getTrackbarPos("min value","tracking")
    uh=cv2.getTrackbarPos("max hue","tracking")
    us=cv2.getTrackbarPos("max sat","tracking")
    uv=cv2.getTrackbarPos("max value","tracking")
    
    lb=np.array([lh,ls,lv])
    ub=np.array([uh,us,uv])
    
    mask=cv2.inRange(img_hsv,lb,ub)
    img_res=cv2.bitwise_and(frame,frame,mask=mask)
    
    cv2.imshow("Video",frame)
    cv2.imshow("mask",mask)
    cv2.imshow("result",img_res)
    cv2.imshow("image hsv",img_hsv)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cap.release()
cv2.destroyAllWindows()    