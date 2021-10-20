# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 17:31:37 2021

@author: DELL
"""

import cv2
import numpy as np

#path="w motors lykan hypersport.jpg"
#img=cv2.resize(cv2.imread(path),(640,480))
 
confThreshold=0.5 #threshold to detect object

cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
   
config_file='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights_file='frozen_inference_graph.pb'

classesFile='coco.names'
classNames=[]
with open(classesFile,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n')
    
#print(classNames)
#print(len(classNames))


#net=cv2.dnn.readNetFromTensorflow(weights_file,config_file)

net=cv2.dnn_DetectionModel(weights_file,config_file)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)


while cap.isOpened():
    success,img=cap.read()
    #net.setInput(cv2.dnn.blobFromImage(img, size=(320, 320), swapRB=True, crop=False))
    #output=net.forward()
    
    classIds,confs,bbox=net.detect(img,confThreshold=0.5)
    #print(classIds)#bbox)
    print(len(bbox))

    if len(classIds)!=0:
        for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.putText(img,str(round(confidence*100,2)),(box[0]+10,box[1]+200),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    
    cv2.imshow("Video",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()