import json
import os
import numpy

files = os.listdir(".")
files = [file for file in files if file.endswith(".txt")]

for file in files:
    with open(file) as reading:

        frames = []
        line_count = -1
        
        for line in reading:
            line_count += 1
            index_angle = line.split(":")

            if len(index_angle) < 2:
                print(index_angle)
                print(line_count)
                break

            angle = eval(index_angle[1])
            if angle == None:
                angle = (69, 420)
            frames.append(numpy.array(angle))

        res = numpy.array(frames)

        numpy.save(file.replace("_rgb_face_gaze_data.txt", "_angles.npy"), res)
