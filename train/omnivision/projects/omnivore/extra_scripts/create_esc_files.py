import os

import numpy as np
from create_audioset_files import read_class_list

DATADIR = "/datasets01/audioset/042319/"
ACTUAL_DATADIR = os.path.join(DATADIR, "data/")
OUTDIR = "/checkpoint/rgirdhar/Work/FB/2021/003_JointImVid/Datasets/ESC-50/"


def main():
    meta = np.loadtxt(
        "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv",
        delimiter=",",
        dtype="str",
        skiprows=1,
    )
    class_ids, classes, human_readable_names = read_class_list(
        "https://raw.githubusercontent.com/YuanGongND/ast/d7d8b4b8e06cdaeb6c843cdb38794c1c7692234c/egs/esc50/data/esc_class_labels_indices.csv"
    )
    # Write it out
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "classes.txt"), "w") as fout:
        fout.write("\n".join(classes))
    np.save(os.path.join(OUTDIR, "label_names.npy"), np.array(human_readable_names))

    for fold in range(1, 5 + 1):
        train_wav_list = []
        train_label_list = []
        eval_wav_list = []
        eval_label_list = []
        for i in range(0, len(meta)):
            cur_label = human_readable_names.index([meta[i][3]])
            cur_path = meta[i][0]
            cur_fold = int(meta[i][1])
            if cur_fold == fold:
                eval_wav_list.append(cur_path)
                eval_label_list.append(cur_label)
            else:
                train_wav_list.append(cur_path)
                train_label_list.append(cur_label)

        print(
            "fold {:d}: {:d} training samples, {:d} test samples".format(
                fold, len(train_wav_list), len(eval_wav_list)
            )
        )

        np.save(os.path.join(OUTDIR, f"fold{fold}_train_filelist.npy"), train_wav_list)
        np.save(os.path.join(OUTDIR, f"fold{fold}_train_labels.npy"), train_label_list)
        np.save(os.path.join(OUTDIR, f"fold{fold}_eval_filelist.npy"), eval_wav_list)
        np.save(os.path.join(OUTDIR, f"fold{fold}_eval_labels.npy"), eval_label_list)


if __name__ == "__main__":
    main()
