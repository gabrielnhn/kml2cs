import json
import os
import numpy

files = os.listdir(".")
files = [file for file in files if file.endswith(".json")]

for file in files:
    with open(file) as reading:
        d = json.load(reading)


        first_label = "openlabel"

        if not first_label in d:
            first_label = "vcd"
        frames = []

        try:
            actions = d[first_label]["actions"]
        except KeyError:
            actions = d[first_label]["actions"]

            print(file)
            exit()

        for action in actions:
            # print(actions[action]["type"])
            if actions[action]["type"] == "gaze_on_road/looking_road":
                for interval in actions[action]["frame_intervals"]:
                    # print(interval)
                    frames.append((interval["frame_start"], interval["frame_end"]))

        total_frames = d[first_label]["streams"]["face_camera"]["stream_properties"]["total_frames"]

            

        ranges = [range(a, b + 1) for (a, b) in frames]

        # with open(file.replace("_rgb_ann_distraction.json", "looking_road_label"), "w"):
        res = numpy.empty(total_frames, dtype=bool)
        for i in range(total_frames):
            res[i] = False
            for frame_range in ranges:
                if i in frame_range:
                    res[i] = True
                    break
        
        numpy.save(file.replace("_rgb_ann_distraction.json", "_looking_road_label.npy"), res)



