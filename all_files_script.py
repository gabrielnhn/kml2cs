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
        '--video_dir',dest='video_dir', help='Video files dir to be processed',
        default=None, type=str)
    
    # parser.add_argument(
    #     '--video_source',dest='video_filename', help='Video file to be processed',
    #     default=None, type=str)

    # parser.add_argument(
    #     '--video_output',dest='video_output', help='Video file output',
    #     default=None, type=str)

    # parser.add_argument(
    #     '--gaze_output',dest='gaze_output', help='Gaze data file output',
    #     default=None, type=str)

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
    video_filename = args.video_dir


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


    all_files = os.listdir(video_dir)
    for file in all_files:
        if file.endswith(".mp4"):
            video_files.append(file)


    ##########
    for video_filename in video_files:

        basename = video_filename.replace(".mp4", "")
        # video_output = basename + ""
        gaze_output = basename + "_gaze_data.txt"

    
        cap = cv2.VideoCapture(video_filename)
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')


        # Check if the webcam is opened correctly
        if not cap.isOpened():
            raise IOError("Cannot open video")

        # video_out = cv2.VideoWriter(video_output, fourcc, 30, (1280,720))

        with open(gaze_output, "w") as output_file:
            with torch.no_grad():
                frame_index = 1
                sucess = True
                success, frame = cap.read()    
                while sucess:
                    area_and_face = []

                    faces = detector(frame)
                    if faces: 
                        for box, landmarks, score in faces:
                            if score < .95:
                                continue

                            x_min=int(box[0])
                            if x_min < 0:
                                x_min = 0
                            y_min=int(box[1])
                            if y_min < 0:
                                y_min = 0
                            x_max=int(box[2])
                            y_max=int(box[3])
                            bbox_width = x_max - x_min
                            bbox_height = y_max - y_min
                            face_area = bbox_height * bbox_width

                            area_and_face.append((face_area, box))

                        # Only process largest face
                        area_and_face.sort()

                        box = area_and_face[0][1]
                        x_min=int(box[0])
                        if x_min < 0:
                            x_min = 0
                        y_min=int(box[1])
                        if y_min < 0:
                            y_min = 0
                        x_max=int(box[2])
                        y_max=int(box[3])
                        bbox_width = x_max - x_min
                        bbox_height = y_max - y_min

                        # Crop image
                        img = frame[y_min:y_max, x_min:x_max]
                        img = cv2.resize(img, (224, 224))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        im_pil = Image.fromarray(img)
                        img=transformations(im_pil)
                        img  = Variable(img).cuda(gpu)
                        img  = img.unsqueeze(0) 
                        
                        # gaze prediction
                        gaze_pitch, gaze_yaw = model(img)
                        
                        pitch_predicted = softmax(gaze_pitch)
                        yaw_predicted = softmax(gaze_yaw)
                        
                        # Get continuous predictions in degrees.
                        pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
                        yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180
                        
                        pitch_predicted= pitch_predicted.cpu().detach().numpy()* np.pi/180.0
                        yaw_predicted= yaw_predicted.cpu().detach().numpy()* np.pi/180.0

                        
                        draw_gaze(x_min,y_min,bbox_width, bbox_height,frame,(pitch_predicted,yaw_predicted),color=(0,0,255))
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)

                        predicted_values = (pitch_predicted,yaw_predicted)
                    else:
                        predicted_values = None


                    output_file.write(f"{frame_index}: {predicted_values}\n")
                    # video_out.write(frame)
                    success, frame = cap.read()    
                    frame_index += 1
