import argparse
import numpy as np
import cv2
import time
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from utils import select_device, draw_gaze
from PIL import Image, ImageOps

from face_detection import RetinaFace
from model import L2CS

import json



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--gpu',dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot.', 
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)
    
    parser.add_argument(
        '--arch',dest='arch',help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    parser.add_argument(
        '--video_dir',dest='video_dir', help='Video files dir to be processed',
        default=None, type=str)

    parser.add_argument(
        '--json_dir',dest='json_dir', help='Json annotations files dir to be processed',
        default=None, type=str)

    parser.add_argument(
        '--xnpys_dir',dest='xnpys_dir', help='X numpy arrays dir',
        default=None, type=str)

    parser.add_argument(
        '--ynpys_dir',dest='ynpys_dir', help='y numpy arrays dir',
        default=None, type=str)


    args = parser.parse_args()
    return args

def getArch(arch,bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS( torchvision.models.resnet.BasicBlock,[2, 2,  2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS( torchvision.models.resnet.BasicBlock,[3, 4,  6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                'The default value of ResNet50 will be used instead!')
        model = L2CS( torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], bins)
    return model

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    arch=args.arch
    batch_size = 1
    gpu = select_device(args.gpu_id, batch_size=batch_size)
    snapshot_path = args.snapshot 
    video_dir = args.video_dir
    json_dir = args.json_dir
    xnpys_dir = args.xnpys_dir
    ynpys_dir = args.ynpys_dir


    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    model = getArch(arch, 90)
    print('Loading snapshot.')
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)
    model.cuda(gpu)
    model.eval()

    softmax = nn.Softmax(dim=1)
    detector = RetinaFace(gpu_id=0)
    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    x=0

    video_files = []
    all_files = os.listdir(video_dir)
    for file in all_files:
        if file.endswith(".mp4"):
            video_files.append(os.path.join(video_dir, file))

    ########## GET GAZE DIRECTION FROM VIDEOS
    # for video_filename in video_files:

    #     basename = video_filename.replace(".mp4", "")
    #     gaze_output = os.path.join(xnpys_dir,  basename + "_gaze_data.npy")
    
    #     cap = cv2.VideoCapture(os.path.join(video_dir, video_filename))
    #     # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    #     # video_out = cv2.VideoWriter(video_output, fourcc, 30, (1280,720))

    #     # Check if the webcam is opened correctly
    #     if not cap.isOpened():
    #         raise IOError(f"Cannot open video {video_filename}")

    #     values = []
    #     with torch.no_grad():
    #         frame_index = 1
    #         sucess = True
    #         success, frame = cap.read()    
    #         while sucess:
    #             area_and_face = []
    #             try:
    #                 faces = detector(frame)
    #             except NotImplementedError:

    #                 values = np.array(values)
    #                 np.save(gaze_output, values)
    #                 break

    #             for box, landmarks, score in faces:
    #                 if score < .95:
    #                     continue

    #                 x_min=int(box[0])
    #                 if x_min < 0:
    #                     x_min = 0
    #                 y_min=int(box[1])
    #                 if y_min < 0:
    #                     y_min = 0
    #                 x_max=int(box[2])
    #                 y_max=int(box[3])
    #                 bbox_width = x_max - x_min
    #                 bbox_height = y_max - y_min
    #                 face_area = bbox_height * bbox_width

    #                 area_and_face.append((face_area, box))

    #                 # Only process largest face
    #                 area_and_face.sort()

    #             if area_and_face:

    #                 box = area_and_face[-1][1]
    #                 x_min=int(box[0])
    #                 if x_min < 0:
    #                     x_min = 0
    #                 y_min=int(box[1])
    #                 if y_min < 0:
    #                     y_min = 0
    #                 x_max=int(box[2])
    #                 y_max=int(box[3])
    #                 bbox_width = x_max - x_min
    #                 bbox_height = y_max - y_min

    #                 # Crop image
    #                 img = frame[y_min:y_max, x_min:x_max]
    #                 img = cv2.resize(img, (224, 224))
    #                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #                 im_pil = Image.fromarray(img)
    #                 img=transformations(im_pil)
    #                 img  = Variable(img).cuda(gpu)
    #                 img  = img.unsqueeze(0) 
                    
    #                 # gaze prediction
    #                 gaze_pitch, gaze_yaw = model(img)
                    
    #                 pitch_predicted = softmax(gaze_pitch)
    #                 yaw_predicted = softmax(gaze_yaw)
                    
    #                 # Get continuous predictions in degrees.
    #                 pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
    #                 yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180
                    
    #                 pitch_predicted= pitch_predicted.cpu().detach().numpy()* np.pi/180.0
    #                 yaw_predicted= yaw_predicted.cpu().detach().numpy()* np.pi/180.0
                    
    #                 draw_gaze(x_min,y_min,bbox_width, bbox_height,frame,(pitch_predicted,yaw_predicted),color=(0,0,255))
    #                 cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)

    #                 predicted_values = (pitch_predicted,yaw_predicted)
    #             else:
    #                 predicted_values = (42, 42) # EXCEPTION VALUES

    #             value = np.array(predicted_values)
    #             values.append(value)
    #             # output_file.write(f"{frame_index}: {predicted_values}\n")
    #             success, frame = cap.read()    


    ######### GET VALUES FROM ANNOTATION JSONs
    basenames = os.listdir(json_dir)
    files = [(os.path.join(json_dir, file), file) for file in basenames if file.endswith(".json")]
    print(files)
    
    for file, basename in files:
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
                raise KeyError

            for action in actions:
                if actions[action]["type"] == "gaze_on_road/looking_road":
                    for interval in actions[action]["frame_intervals"]:
                        frames.append((interval["frame_start"], interval["frame_end"]))

            total_frames = d[first_label]["streams"]["face_camera"]["stream_properties"]["total_frames"]
            ranges = [range(a, b + 1) for (a, b) in frames]

            res = np.empty(total_frames, dtype=bool)
            for i in range(total_frames):
                res[i] = False
                for frame_range in ranges:
                    if i in frame_range:
                        res[i] = True
                        break
            
            out_file = basename.replace("_rgb_ann_distraction.json", "_looking_road_label.npy")
            out_file = os.path.join(ynpys_dir, out_file) 
            np.save(out_file, res)


    ###### JOIN ARRAYS
    # get order according to json dir
    prefixes = [file.replace("_rgb_ann_distraction.json", "") for _, file in files] 
    order = prefixes
    print(order)

    ### JOIN X ARRAYS

    sufix = "_rgb_face.mp4"

    all_arrays_x = []
    for prefix in order:
        array_file = os.path.join(xnpys_dir, prefix + sufix) 
        all_arrays_x += list(np.load(array_file, allow_pickle=True))

    all_arrays_x = np.array(all_arrays_x)
    np.save("ALL_FILES_X.npy", os.path.join(xnpys_dir, all_arrays_x))

    ### JOIN y ARRAYS

    sufix = "_rgb_face.mp4"

    all_arrays_y = []
    for prefix in order:
        array_file = os.path.join(ynpys_dir, prefix + sufix) 
        all_arrays_y += list(np.load(array_file, allow_pickle=True))

    all_arrays_y = np.array(all_arrays_y)
    np.save("ALL_FILES_y.npy", os.path.join(ynpys_dir, all_arrays_y))

