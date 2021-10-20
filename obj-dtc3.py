# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 18:22:13 2021

@author: DELL
"""

import cv2
import numpy as np

cap=cv2.VideoCapture(0)
whT=250 #keep it between 220 and 320 for better frame rate
hgT=250
confidenceThreshold=0.5
nmsThreshold=0.3 #lower ->more aggressive and less no of boxes 

classesFile='coco.names' #file containing the names of classes in coco dataset
classNames=[] #80 different names

with open(classesFile,'rt') as f: 
    classNames=f.read().rstrip('\n').split('\n')
#print(classNames)
modelConfiguration='yolov3.cfg' #modelConfiguration='yolov3-tiny.cfg'
modelWeights='yolov3.weights'   #modelWeights='yolov3-tiny.weights'

net=cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights) #creating network
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


cap.set(3,640)
cap.set(4,480)
#cap.set(10,100) #brightness


def findObjects(outputs,img):
    ht,wt,ct=img.shape #height, width and channels of the image
    boundingBox=[] #it'll contain values of x,y,width and height
    classIds=[]
    confs=[]
    
    for output in outputs: # 3 outputs
        for det in output:
            scores=det[5:]
            classID=np.argmax(scores) #find index of max value
            confidence=scores[classID] #get the max value/probability
            if confidence>confidenceThreshold:
                w,h=int(det[2]*wt), int(det[3]*ht) #multipyling with width and height of image to get the pixel value 
                x,y=int((det[0]*wt)-w/2),int((det[1]*ht)-h/2)
                boundingBox.append([x,y,w,h])
                classIds.append(classID)
                confs.append(float(confidence))
    print(len(boundingBox))
    indices=cv2.dnn.NMSBoxes(boundingBox,confs,confidenceThreshold,nmsThreshold) #non-max suppression function - elimnates overlapping boxes
    
    for i in indices:
        i=i[0]
        box=boundingBox[i]
        x,y,w,h=box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,0,255),2)

while cap.isOpened():
    timer=cv2.getTickCount()
    success,img=cap.read() 
    #path="w motors lykan hypersport.jpg"
    #img=cv2.resize(cv2.imread(path),(640,480))
    
    #path="1.jpg"
    #img=cv2.resize(cv2.imread(path),(1280,720))
    
    fps=cv2.getTickFrequency()/(cv2.getTickCount()-timer)
    cv2.putText(img,str(int(fps)),(75,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    
    blob=cv2.dnn.blobFromImage(img,1/255,(whT,hgT),[0,0,0],1,crop=False) #converting image to blob (blob - type of format of image accepted by the network)
    net.setInput(blob)
     
    layerNames=net.getLayerNames()
    #print(layerNames)
    outputNames=[layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    #print(outputNames)
    #print(net.getUnconnectedOutLayers())
    
    outputs=net.forward(outputNames) #send the image as forward pass to the network
    #print(outputs[0].shape)
    #print(outputs[1].shape)
    #print(outputs[2].shape)
    #print(outputs[0][0])
    
    findObjects(outputs,img)
    
    cv2.imshow("Video",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows() 