import json
import os
from cProfile import label

import numpy as np


def process_food101():
    splits = ["train", "test"]
    root_dir = "/datasets01/food-101/020722/"
    dst_dir = "/checkpoint/imisra/datasets/food101"

    for split in splits:
        data_file = os.path.join(root_dir, "meta", f"{split}.json")
        with open(data_file, "r") as fh:
            data = json.load(fh)
        if split == "train":
            classnames = list(data.keys())
            classnames.sort()
            classname_to_idx = {classnames[x]: x for x in range(len(classnames))}
        else:
            classnames_val = list(data.keys())
            classnames_val.sort()
            assert len(classnames_val) == len(classnames)

        image_list = []
        label_list = []
        for _, classname in enumerate(classnames):
            filenames = [
                os.path.join(root_dir, "images", x + ".jpg") for x in data[classname]
            ]
            image_list.extend(filenames)
            label_list.extend([classname_to_idx[classname]] * len(filenames))

        assert os.path.isfile(image_list[0]), f"{image_list[0]} not found"
        image_list = np.array(image_list)
        label_list = np.array(label_list, dtype=np.int64)
        assert label_list.min() == 0 and label_list.max() == (len(classnames) - 1)
        np.save(os.path.join(dst_dir, f"{split}_image_names.npy"), image_list)
        np.save(os.path.join(dst_dir, f"{split}_labels.npy"), label_list)

    classnames_zs = [x.replace("_", " ") for x in classnames]
    classnames_zs = np.array(classnames_zs)[..., None]
    np.save(os.path.join(dst_dir, "classnames_zs.npy"), classnames_zs)

    templates = np.load("/checkpoint/imisra/datasets/in1k_disk/templates_openai.npy")
    templates = [x.replace(".", ", a type of food.") for x in templates]

    np.save(os.path.join(dst_dir, "templates_openai_for_food.npy"), templates)


def process_places365():
    splits = ["train_standard", "val"]
    root_dir = "/datasets01/Places365/120221/"
    dst_dir = "/checkpoint/imisra/datasets/places365"
    image_dirs = {
        "train_standard": "data_large",
        "val": "val_large",
    }

    with open(os.path.join(root_dir, "categories_places365.txt")) as fh:
        data = fh.readlines()
        data = [x.strip() for x in data]
        classnames = []
        idx_to_classnames = {}
        for line in data:
            line = line.split("/")[2:]
            line = " ".join(line)
            dt = line.split(" ")
            cls_idx = int(dt[-1])
            cls = " ".join(dt[:-1])
            classnames.append(cls)
            idx_to_classnames[cls_idx] = cls

    for split in splits:
        split_file = os.path.join(root_dir, "places365_" + split + ".txt")
        with open(split_file, "r") as fh:
            data = fh.readlines()

        data = [x.strip() for x in data]
        image_list = []
        label_list = []
        for line in data:
            line = line.split(" ")
            if split == "val":
                basename = line[0]
            else:
                basename = line[0][1:]
            image_name = os.path.join(root_dir, image_dirs[split], basename)
            image_list.append(image_name)
            label_list.append(int(line[1]))

        assert os.path.isfile(image_list[0]), f"{image_list[0]} not found"
        image_list = np.array(image_list)
        label_list = np.array(label_list, dtype=np.int64)
        assert label_list.min() == 0 and label_list.max() == (len(classnames) - 1)
        np.save(os.path.join(dst_dir, f"{split}_image_names.npy"), image_list)
        np.save(os.path.join(dst_dir, f"{split}_labels.npy"), label_list)

    classnames_zs = [x.replace("_", " ") for x in classnames]
    classnames_zs = np.array(classnames_zs)[..., None]
    np.save(os.path.join(dst_dir, "classnames_zs.npy"), classnames_zs)


def process_pets():
    root_dir = "/datasets01/Pets/071222/"
    dst_dir = "/checkpoint/imisra/datasets/pets"

    splits = ["trainval", "test"]
    for split in splits:
        split_file = os.path.join(root_dir, "annotations", f"{split}.txt")
        with open(split_file, "r") as fh:
            data = fh.readlines()
        data = [x.strip() for x in data]

        image_list = []
        label_list = []
        idx_to_classnames = {}
        classnames = []
        for line in data:
            dt = line.split(" ")
            cls = dt[0]
            cls_idx = int(dt[1])
            classname = " ".join(cls.split("_")[:-1])
            label_list.append(cls_idx)
            image_list.append(os.path.join(root_dir, "images", cls + ".jpg"))
            if cls_idx not in idx_to_classnames:
                idx_to_classnames[cls_idx] = classname
                classnames.append(classname)
            else:
                assert idx_to_classnames[cls_idx] == classname

        assert os.path.isfile(image_list[0]), f"{image_list[0]} not found"
        label_list = np.array(label_list, dtype=np.int64)
        label_list = label_list - 1
        image_list = np.array(image_list)
        assert label_list.min() == 0 and label_list.max() == (len(classnames) - 1)
        np.save(os.path.join(dst_dir, f"{split}_image_names.npy"), image_list)
        np.save(os.path.join(dst_dir, f"{split}_labels.npy"), label_list)

    classnames_zs = [x.replace("_", " ") for x in classnames]
    classnames_zs = np.array(classnames_zs)[..., None]
    np.save(os.path.join(dst_dir, "classnames_zs.npy"), classnames_zs)

    templates = np.load("/checkpoint/imisra/datasets/in1k_disk/templates_openai.npy")
    templates = [x.replace(".", ", a type of pet.") for x in templates]

    np.save(os.path.join(dst_dir, "templates_openai_for_pets.npy"), templates)


if __name__ == "__main__":
    process_food101()
    process_places365()
    process_pets()
