import os
import numpy as np
import pandas as pd
import cv2
import pyautogui # https://youtu.be/qIJpBz6R2Uw

video_cap = cv2.VideoCapture('Color Reference.mp4')
location = input('Which location do you want to check RGB value for? ')
location1, location2 = location.split(',')
while video_cap.isOpened():
    _, frame = video_cap.read()
    # VideoCapture read frames in BGR order
    RGB_frame = frame[:, :, ::-1]
    RGB_value = RGB_frame[int(location1):int(location2), int(location1):int(location2)]
    mean_RGB_value = np.mean(RGB_value, axis = (0, 1))
    text = f"RGB value at ({location1}, {location2}) is {(mean_RGB_value)}."
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame by frame
    '''
    Compression: Most video formats use some form of lossy compression. 
    When a video is compressed, some data is lost, which can lead to differences in pixel values when the video is read back.
    '''
    cv2.imshow('video frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

video_cap.release()
cv2.destroyAllWindows()