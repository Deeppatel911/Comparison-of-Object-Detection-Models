import numpy as np
import time
import cv2
import pixellib
from pixellib.instance import instance_segmentation
segment_image = instance_segmentation()
segment_image.load_model("mask_rcnn_coco.h5") 
camera = cv2.VideoCapture(0)

# used to record the time when we processed last frame
prev_frame_time = 0
 
# used to record the time at which we processed current frame
new_frame_time = 0

while camera.isOpened():
    res,frame=camera.read()
    ### Apply Segmentation
    result=segment_image.segmentFrame(frame,show_bboxes=True)
    image=result[1]
    cv2.imshow('Image Segmentation',image)


    # Our operations on the frame come here
    gray = frame
 
    # resizing the frame size according to our need
    gray = cv2.resize(gray, (500, 300))



    # font which we will be using to display FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for this frame
    new_frame_time = time.time()
 
    # Calculating the fps
 
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
 
    # converting the fps into integer
    fps = int(fps)
 
    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)
 
    # putting the FPS count on the frame
    cv2.putText(gray, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
 
    # displaying the frame with fps
    cv2.imshow('frame', gray)

    if cv2.waitKey(5) & 0xFF==ord('q'):
        break

    

camera.release()
cv2.destroyAllWindows()






