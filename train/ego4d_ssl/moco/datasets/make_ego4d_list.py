"""
Use this file to create a txt file with path to all the frames
"""

import os

BASE_DIR = "/private/home/aravraj/r3m_data/ego4d/"
FILE_LIST = []
l1_dirs = os.listdir(BASE_DIR)
for l1 in l1_dirs:
    l1_full = os.path.join(BASE_DIR, l1)
    if os.path.isdir(l1_full):
        l2_dirs = os.listdir(l1_full)
        for l2 in l2_dirs:
            l2_full = os.path.join(l1_full, l2)
            if os.path.isdir(l2_full):
                l3_files = os.listdir(l2_full)
                for l3 in l3_files:
                    if l3.endswith(".jpg"):
                        # image
                        FILE_LIST.append(os.path.join(l2_full, l3))

print("Number of frames in the dataset: %i" % len(FILE_LIST))

with open("ego4d_r3m.txt", "w") as filehandle:
    # for img in FILE_LIST[:100000]:
    for img in FILE_LIST:
        filehandle.write("%s\n" % img)
