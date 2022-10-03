import numpy as np
from iopath.common.file_io import g_pathmgr


DATA_DIR = "/checkpoint/mannatsingh/datasets/places365"
OUT_DIR = "/checkpoint/mannatsingh/datasets/omniscale/places365/"
TRAIN_SUBDIR = "data_large"
VAL_SUBDIR = "val_large"
CATEGORY_FILE = "categories_places365.txt"
VAL_FILE = "places365_val.txt"
CLASSES = 365


def main():
    # classes
    class_map = {}
    with g_pathmgr.open(f"{DATA_DIR}/{CATEGORY_FILE}") as f:
        for line in f.readlines():
            cls_name, cls_label = line.split()
            class_map[cls_name] = int(cls_label)
    print(class_map)

    # train split
    # places train data is structured as
    # "data_large/a/art_gallery/file.jpg"
    # "data_large/a/apartment_building/outdoor/file.jpg"
    # "data_large/b/ballroom/file.jpg"
    train_images = []
    train_labels = []
    for cls_path in sorted(class_map.keys()):
        files = sorted(g_pathmgr.ls(f"{DATA_DIR}/{TRAIN_SUBDIR}{cls_path}"))
        for f in files:
            assert f.endswith("jpg"), (cls_path, f)
            train_images.append(f"{TRAIN_SUBDIR}/{cls_path[1:]}/{f}")
            train_labels.append(class_map[cls_path])
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    np.save(OUT_DIR + "train_images.npy", train_images)
    np.save(OUT_DIR + "train_labels.npy", train_labels)
    print(len(train_images), train_images)
    print(len(train_labels), train_labels)

    # val images
    # places val data is structured as
    # "val_large/file.jpg"
    # the classes are stored in the val file
    val_images = []
    val_labels = []
    with g_pathmgr.open(f"{DATA_DIR}/{VAL_FILE}") as f:
        for line in f.readlines():
            img_file, label = line.split()
            val_images.append(f"{VAL_SUBDIR}/{img_file}")
            val_labels.append(int(label))
            assert g_pathmgr.exists(f"{DATA_DIR}/{VAL_SUBDIR}/{img_file}")
    val_images = np.array(val_images)
    val_labels = np.array(val_labels)
    np.save(OUT_DIR + "val_images.npy", val_images)
    np.save(OUT_DIR + "val_labels.npy", val_labels)
    print(len(val_images), val_images)
    print(len(val_labels), val_labels)


if __name__ == "__main__":
    main()
