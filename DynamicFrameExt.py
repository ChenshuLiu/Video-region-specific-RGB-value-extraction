import os
import numpy as np
import pandas as pd
import cv2
import pyautogui # https://youtu.be/qIJpBz6R2Uw

video_cap = cv2.VideoCapture('Video-region-specific-RGB-value-extraction/Color Reference.mp4')
ret, first_frame = video_cap.read()
if ret:
    points = []
    def select_point(event, x, y, flags, param):
        global points
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(temp_frame, (x, y), 2, (0, 0, 255), -1)
            cv2.imshow('Image', temp_frame)
            points.append([x, y])
    temp_frame = first_frame.copy()
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', select_point)
    while True:
        cv2.imshow('Image', temp_frame)
        if cv2.waitKey(0) & 0xFF == 13:  # return key
            break
    # Convert points to numpy array
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    mask = np.zeros(first_frame.shape[:2], dtype = np.uint8)
    cv2.fillPoly(mask, [pts], (255, 255, 255))
    cv2.waitKey(0)

    while True:
        _, frame = video_cap.read()
        RGB_mean = []
        for i in range(3):
            channel_values = frame[:, :, i][mask == 255]
            RGB_mean.append(np.mean(channel_values))
        blue, green, red = RGB_mean
        text = f"Red channel is {red}, Green channel is {green}, Blue channel is {blue}"
        for vertex_id in range(pts.shape[0]):
            vertex = tuple(pts[vertex_id, :, :][0])
            cv2.circle(frame, vertex, 2, (0, 0, 255), -1)
        cv2.polylines(frame, [pts], isClosed = True, color = (0, 0, 255), thickness = 2)
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