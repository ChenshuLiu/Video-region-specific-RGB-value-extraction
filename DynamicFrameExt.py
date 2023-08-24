import os
import numpy as np
import pandas as pd
import cv2
import pyautogui # https://youtu.be/qIJpBz6R2Uw

video_cap = cv2.VideoCapture('Video-region-specific-RGB-value-extraction/Color Reference.mp4')
'''
location = input('Which location do you want to check RGB value for? ')
location1, location2 = location.split(',')
'''
ret, first_frame = video_cap.read()
if ret:
    # List to store points (vertices)
    points = []

    # detection of mouse clicking event
    def select_point(event, x, y, flags, param):
        global points
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(temp_frame, (x, y), 2, (0, 0, 255), -1)  # Red dot for visualization
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

    cv2.waitKey(0)

    while True:
        _, frame = video_cap.read()
        # VideoCapture read frames in BGR order
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [pts], (255, 255, 255))
        polygonal_region = cv2.bitwise_and(frame, mask)
        RGB_polygonal_region = polygonal_region[:, :, ::-1]
        mean_RGB_value = np.mean(RGB_polygonal_region, axis = (0, 1))
        text = f"Mean RGB value is {(mean_RGB_value)}."
        for vertex_id in range(pts.shape[0]):
            vertex = tuple(pts[vertex_id, :, :][0])
            print(vertex)
            print(type(vertex))
            cv2.circle(frame, vertex, 2, (0, 0, 255), -1)
        '''
        for i in range(1, pts.shape[0]-1):
            vtx_1 = tuple(pts[i-1, :, :][0])
            vtx_2 = tuple(pts[i, :, :][0])
            vtx_3 = tuple(pts[i+1, :, :][0])
            cv2.line(frame, vtx_1, vtx_2, (0, 0, 255), 2)
            cv2.line(frame, vtx_2, vtx_3, (0, 0, 255), 2)
        '''
        cv2.polylines(frame, [pts], isClosed = True, color = (0, 0, 255), thickness = 2)
        #cv2.circle(frame, pts, 2, (0, 0, 255), -1)
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