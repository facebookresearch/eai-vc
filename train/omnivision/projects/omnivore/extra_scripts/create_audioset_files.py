import json
import os

import numpy as np
import pandas

DATADIR = "/datasets01/audioset/042319/"
ACTUAL_DATADIR = os.path.join(DATADIR, "data/")
CSV_FILES = [
    "balanced_train_segments.csv",
    "eval_segments.csv",
    # comment it out if not changing any
    "unbalanced_train_segments.csv",
]
if 0:
    DATATYPEDIR = "audio"
    DATATYPEEXT = ".flac"
    OUTDIR = "/checkpoint/rgirdhar/Work/FB/2021/003_JointImVid/Datasets/AudioSet/"
elif 1:
    DATATYPEDIR = "video"
    DATATYPEEXT = ".mp4"
    OUTDIR = "/checkpoint/rgirdhar/Work/FB/2021/003_JointImVid/Datasets/AudioSetVideo/"


def read_class_list(url):
    cls_names = pandas.read_csv(url)
    return (
        cls_names.loc[:, "index"].tolist(),
        cls_names.loc[:, "mid"].tolist(),
        # Making list of list since that is how we do it for imagenet etc
        # Not splitting the string into multiple items which would refer to
        # the same sound since 1) not clear how to split (comma doesn't always
        # precede a synonym), and 2) need to have same num of items for each
        # class for batching reasons..
        [[el] for el in cls_names.loc[:, "display_name"].tolist()],
    )


def split_label(s):
    return s.strip('"').split(",")


def read_files(file_list):
    data = {}
    for csv_fpath in file_list:
        # remove the ".csv" from the name for key
        data[csv_fpath[:-4]] = pandas.read_csv(
            os.path.join(DATADIR, csv_fpath),
            comment="#",
            sep=", ",
            header=None,
        )
    return data


def save_ast_style(paths, labels, outfpath):
    data = []
    for path, label in zip(paths, labels):
        data.append({"wav": os.path.join(DATADIR, "data", path), "labels": label})
    with open(outfpath, "w") as fout:
        json.dump({"data": data}, fout)


def main():
    # Reading from the AST codebase for consistency,
    # and textual names of the classes
    class_ids, classes, human_readable_names = read_class_list(
        "https://raw.githubusercontent.com/YuanGongND/ast/d7d8b4b8e06cdaeb6c843cdb38794c1c7692234c/egs/audioset/data/class_labels_indices.csv"
    )
    # Write it out
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "classes.txt"), "w") as fout:
        fout.write("\n".join(classes))
    np.save(os.path.join(OUTDIR, "label_names.npy"), np.array(human_readable_names))

    # Load all csv files
    data = read_files(CSV_FILES)

    # Write out the .npy files for each
    for key, table in data.items():
        fpaths = []
        # Storing the list of labels is a complex problem
        # Tried a list of integers. Unfortuanately numpy can only treat them as object arrays which
        # makes shared mem loading hard.
        # Tried creating 1-hot vectors directly here. Unfortunately numpy boolean arrays still use
        # 1 byte itemsize for each element, so those arrays become large
        # So going with a comma separated string, as that seems to work with shared mem loading,
        # and also should be possible to convert back to list in label transforms.
        labels_found = set()
        labels = []
        labels_strings = []
        for _, row in table.iterrows():
            start_msec = int(row[1] * 1000)
            end_msec = int(row[2] * 1000)
            data_path = (
                f"{key}/{DATATYPEDIR}/{row[0]}_{start_msec}_{end_msec}{DATATYPEEXT}"
            )
            if not os.path.exists(os.path.join(ACTUAL_DATADIR, data_path)):
                continue
            labels_strings.append(row[3].strip('"'))
            this_labels = [class_ids[classes.index(el)] for el in split_label(row[3])]
            fpaths.append(data_path)
            labels.append(",".join([str(el) for el in this_labels]))
            labels_found = labels_found.union(set(this_labels))
        np.save(os.path.join(OUTDIR, f"{key}_filelist.npy"), fpaths)
        np.save(os.path.join(OUTDIR, f"{key}_labels.npy"), labels)
        save_ast_style(fpaths, labels_strings, os.path.join(OUTDIR, f"{key}_ast.json"))
        perc_data_found = (len(labels) / len(table)) * 100.0
        perc_labels_found = (len(labels_found) / len(classes)) * 100.0
        print(
            f"Done for {key}. Found {perc_data_found:0.02f}% "
            f"clips and {perc_labels_found:0.02f}% labels."
        )


if __name__ == "__main__":
    main()
