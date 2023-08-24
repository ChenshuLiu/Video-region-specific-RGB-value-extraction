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

#######################################################################################

import cv2
import numpy as np

# List to store points (vertices)
points = []

# detection of mouse clicking event
def select_point(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(temp_img, (x, y), 2, (0, 0, 255), -1)  # Red dot for visualization
        cv2.imshow('Image', temp_img)
        points.append([x, y])

image_path = 'Video-region-specific-RGB-value-extraction/Color Palette.png'
image = cv2.imread(image_path)
# modifications will be done on the temp image
temp_img = image.copy()

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', select_point)

# continuously display temp_img in the newWindow
# escape when the ESC key is called
while True:
    cv2.imshow('Image', temp_img)
    if cv2.waitKey(20) & 0xFF == 27:  # ESC key
        break

cv2.destroyAllWindows()

# Convert points to numpy array
pts = np.array(points, np.int32)
pts = pts.reshape((-1, 1, 2))

# Create a mask of zeros (black)
mask = np.zeros_like(image)

# Fill the polygon in the mask with white
cv2.fillPoly(mask, [pts], (255, 255, 255))

# Extract the polygonal region from the image
polygonal_region = cv2.bitwise_and(image, mask)
RGB = polygonal_region[:, :, ::-1]
mean_RGB_value = np.mean(RGB, axis = (0, 1))
print(f"The mean RGB value is {mean_RGB_value}")
# Display the extracted region
cv2.imshow('Extracted Region', polygonal_region)
cv2.waitKey(0)
cv2.destroyAllWindows()
