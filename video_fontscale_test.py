import argparse
import numpy as np
import cv2
import time

video_filename = "/home/gnhn/DMD/new_data/gA_1_s5_2019-03-14T14;26;17+01;00_rgb_face.mp4"
print(video_filename)
cap = cv2.VideoCapture(video_filename)
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open video")

video_out = cv2.VideoWriter("vid.mp4", fourcc, 30, (1280,720))

frame_index = 1
sucess = True
success, frame = cap.read()    
while sucess:
    
    prediction = True
    if prediction:
        output_str = f"Looking at road elements"
        color = (0, 255, 100)
    else:
        output_str = f"Distracted"
        color = (0, 100, 255)

    text_size, _ = cv2.getTextSize(output_str, cv2.FONT_HERSHEY_PLAIN, 4, 4)
    text_w, text_h = text_size

    cv2.putText(frame, output_str, (50, 20 + text_h), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)


    # output_file.write(f"{frame_index}: {predicted_values}\n")
    video_out.write(frame)
    success, frame = cap.read()    
    frame_index += 1
