# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 17:17:55 2021

@author: DELL
"""

import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd='C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
confg=r'--oem 3 --psm 6 outputbase digits'

img=cv2.resize(cv2.imread('2.png'),(640,480))
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#print(pytesseract.image_to_string(img))

#Dectecting characters
# hImg,wImg,_=img.shape
# boxes=pytesseract.image_to_boxes(img)
# for b in boxes.splitlines():
#     b=b.split(' ')
#     print(b)
#     x,y,w,h=int(b[1]),int(b[2]),int(b[3]),int(b[4])
#     cv2.rectangle(img, (x,hImg-y), (w,hImg-h), (0,0,255),1)
#     cv2.putText(img, b[0], (x,hImg-y+25), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255))

#Dectecting words
hImg,wImg,_=img.shape
boxes=pytesseract.image_to_data(img)
#print(pytesseract.image_to_data(img))
for x,b in enumerate(boxes.splitlines()): #counter
    if x!=0:
        b=b.split()
        print(b)
        if len(b)==12:
            x,y,w,h=int(b[6]),int(b[7]),int(b[8]),int(b[9])
            cv2.rectangle(img, (x,y), (w+x,h+y), (0,0,255),1)
            cv2.putText(img, b[11], (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255))

# #Dectecting only digits
# hImg,wImg,_=img.shape
# boxes=pytesseract.image_to_data(img,config=confg)
# print(pytesseract.image_to_data(img))
# for x,b in enumerate(boxes.splitlines()): #counter
#     if x!=0:
#         b=b.split()
#         print(b)
#         if len(b)==12:
#             x,y,w,h=int(b[6]),int(b[7]),int(b[8]),int(b[9])
#             cv2.rectangle(img, (x,y), (w+x,h+y), (0,0,255),1)
#             cv2.putText(img, b[11], (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255))


cv2.imshow('image', img)
cv2.waitKey(0)

