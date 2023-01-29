import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import tensorflow as tf
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates
from detect_file import Detect_final

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


def handle_left_click(event, x, y, flags, points):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])

def draw_polygon (frame, points):
    for point in points:
        frame = cv2.circle( frame, (point[0], point[1]), 5, (0,0,255), -1)  
    frame = cv2.polylines(frame, [np.int32(points)], False, (255,0, 0), thickness=2)
    return frame


start_detect = False
model = Detect_final()
points = []
Ptime = 0

video_path = "video/wakeup_3.mp4"
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    if not ret:
        print("Ignoring empty camera frame.")
    # If loading a video, use 'break' instead of 'continue'.
        break
    
    if start_detect:
        frame = model.output(frame = frame, points = points)
    
    cTime = time.time()
    fps =1/(cTime-Ptime)
    Ptime = cTime 

    #cv2.putText(frame ,f'FPS:{int(fps)}',(400,70),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,0),3)    
    
    #Ve ploygon
    frame = draw_polygon(frame, points)
    key = cv2.waitKey(10)
    
    if key == ord('q'):
        #print(points)q
        break
    elif key == ord('d'):
        points.append(points[0])
        start_detect = True
        
    cv2.imshow("Baby Monitor", frame)
    
    cv2.setMouseCallback('Baby Monitor', handle_left_click, points)

cap.release()
cv2.destroyAllWindows()