import numpy as np


# IN_DIR = "/checkpoint/imisra/datasets/inat18/"
# IN_FILES = [
#     "train_images.npy",
#     "val_images.npy",
# ]
# OUT_DIR = "/checkpoint/mannatsingh/datasets/omniscale/inat18/"
# PREFIX = "/checkpoint/htouvron/inat_dataset/train_val2018/"

# IN_DIR = "/checkpoint/imisra/datasets/in1k_disk/"
# IN_FILES = [
#     "train_images_global.npy",
#     "val_images_global.npy",
# ]
# OUT_DIR = "/checkpoint/mannatsingh/datasets/omniscale/in1k/"
# PREFIX = "/datasets01/imagenet_full_size/061417/"

# IN_DIR = "/checkpoint/rgirdhar/Work/FB/2021/003_JointImVid/Datasets/SSv2/"
# IN_FILES = [
#     "vidpaths_train.npy",
#     "vidpaths_validation.npy",
# ]
# OUT_DIR = "/checkpoint/mannatsingh/datasets/omniscale/ssv2/"
# PREFIX = "/datasets01/SSV2/videos/"

IN_DIR = "/checkpoint/mannatsingh/datasets/kinetics_400_manifold_meta/"
IN_FILES = [
    "vidpaths_train.npy",
    "vidpaths_val.npy",
]
OUT_DIR = "/checkpoint/mannatsingh/datasets/omniscale/k400/"
PREFIX = "manifold://omnivore/tree/dataset/kinetics400_high_qual_320_trimmed/"


def main():
    for file in IN_FILES:
        data = np.load(f"{IN_DIR}/{file}")
        out_data = []
        for row in data:
            out_row = row.replace(PREFIX, "")
            assert len(row) == len(PREFIX) + len(out_row)
            out_data.append(out_row)
        out = np.array(out_data)
        print(data.dtype)
        print(data)
        print(out.dtype)
        print(out)
        np.save(f"{OUT_DIR}/{file}", out)


if __name__ == "__main__":
    main()
