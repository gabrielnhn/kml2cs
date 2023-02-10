import os
import numpy as np

train = [
    "gA_1_s1_2019-03-08T09;31;15+01;00",
    "gA_2_s1_2019-03-08T10;01;44+01;00",
    "gA_3_s1_2019-03-08T10;27;38+01;00",
    "gA_4_s1_2019-03-13T10;36;15+01;00",
    "gA_5_s1_2019-03-08T10;57;00+01;00",
    "gB_6_s1_2019-03-11T13;55;14+01;00",
    "gB_7_s1_2019-03-11T14;22;01+01;00",
    "gB_8_s1_2019-03-11T15;01;33+01;00",
    "gB_9_s1_2019-03-07T16;36;24+01;00",
    "gB_10_s1_2019-03-11T15;24;54+01;00",
    "gC_11_s1_2019-03-04T09;33;18+01;00",
    "gC_12_s1_2019-03-13T10;23;45+01;00",
    "gC_13_s1_2019-03-04T10;26;12+01;00",
    "gC_14_s1_2019-03-04T11;56;20+01;00",
    "gC_15_s1_2019-03-04T11;24;57+01;00",
    "gF_21_s1_2019-03-05T09;48;30+01;00",
    "gF_22_s1_2019-03-04T14;54;55+01;00",
    "gF_23_s1_2019-03-04T16;21;10+01;00",
    "gF_24_s1_2019-03-04T15;26;10+01;00",
    "gF_25_s1_2019-03-04T15;53;22+01;00",
]
test = [
    "gE_26_s1_2019-03-15T09;25;24+01;00",
    "gE_27_s1_2019-03-07T13;18;37+01;00",
    "gE_28_s1_2019-03-15T10;23;30+01;00",
    "gE_29_s1_2019-03-15T13;58;00+01;00",
    "gE_30_s1_2019-03-15T10;58;06+01;00",
    "gZ_31_s1_2019-04-08T09;48;48+02;00",
    "gZ_32_s1_2019-04-08T12;01;26+02;00",
    "gZ_33_s1_2019-04-08T10;08;19+02;00",
    "gZ_34_s1_2019-04-08T12;25;28+02;00",
]

xnpys_dir = "/home/gnhn19/new_training_s1/x"
ynpys_dir = "/home/gnhn19/new_training_s1/y"



### JOIN X ARRAYS
sufix = "_rgb_face_gaze_data.npy"

#Train

all_arrays_x = []
for prefix in train:
    array_file = os.path.join(xnpys_dir, prefix + sufix) 
    all_arrays_x += list(np.load(array_file, allow_pickle=True))

all_arrays_x = np.array(all_arrays_x)
np.save(os.path.join(xnpys_dir, "TRAIN_X.npy"), all_arrays_x)

#Test

all_arrays_x = []
for prefix in test:
    array_file = os.path.join(xnpys_dir, prefix + sufix) 
    all_arrays_x += list(np.load(array_file, allow_pickle=True))

all_arrays_x = np.array(all_arrays_x)
np.save(os.path.join(xnpys_dir, "TEST_X.npy"), all_arrays_x)


### JOIN y ARRAYS
sufix = "_looking_road_label.npy"

#Train

all_arrays_y = []
for prefix in train:
    array_file = os.path.join(ynpys_dir, prefix + sufix) 
    all_arrays_y += list(np.load(array_file, allow_pickle=True))

all_arrays_y = np.array(all_arrays_y)
np.save(os.path.join(ynpys_dir, "TRAIN_y.npy"), all_arrays_y)

#Test

all_arrays_y = []
for prefix in test:
    array_file = os.path.join(ynpys_dir, prefix + sufix) 
    all_arrays_y += list(np.load(array_file, allow_pickle=True))

all_arrays_y = np.array(all_arrays_y)
np.save(os.path.join(ynpys_dir, "TEST_y.npy"), all_arrays_y)

