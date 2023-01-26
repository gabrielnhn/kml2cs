import os
import numpy as np


all_files = os.listdir(".")

npfiles = [file for file in all_files if file.endswith(".npy")]


all_arrays = []
for npfile in npfiles:
    all_arrays += list(np.load(npfile, allow_pickle=True))

all_arrays = np.array(all_arrays)
np.save("ALL_FILES.npy", all_arrays)

###
import os
import numpy as np

a = np.load("ALL_FILES.npy", allow_pickle=True)
print(a)


###
import os
import numpy as np


all_files = os.listdir(".")
npfiles = [file for file in all_files if file.endswith(".npy")]

all_arrays = []
for npfile in npfiles:
    if len(np.load(npfile, allow_pickle=True).shape) != 2:
        print(npfile, "\n\n")
        print(np.load(npfile, allow_pickle=True))

