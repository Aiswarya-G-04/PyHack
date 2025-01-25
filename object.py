import cv2
import cvlib as cv
from cvlib.object.detection import draw_bbox
from gtts import playground
video=cv2.VideoCapture(1)
while True:
    ret,frame=video.read()
    bbox,frame,cont=cv.detect_common_objects(frame)
    output_imagedraw_bbox(frame,bbox,label,conf)
    cv.imshow("object detection",output_image)
    if cv2.waitKey(1)