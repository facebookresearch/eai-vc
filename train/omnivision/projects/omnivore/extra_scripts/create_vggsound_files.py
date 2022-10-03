import os

import numpy as np
import pandas
from tqdm import tqdm

DATADIR = "/fsx-omnivore/rgirdhar/data/VGGSound/wav_24k/"
OUTDIR = "/checkpoint/rgirdhar/Work/FB/2021/003_JointImVid/Datasets/VGGSound/"


def main():
    data = pandas.read_csv(
        "https://www.robots.ox.ac.uk/~vgg/data/vggsound/vggsound.csv",
        header=None,
    )
    human_readable_cnames = sorted(set(data.iloc[:, 2].tolist()))
    assert len(human_readable_cnames) == 309
    classes = [el.replace(" ", "_") for el in human_readable_cnames]
    # Write it out
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "classes.txt"), "w") as fout:
        fout.write("\n".join(classes))
    np.save(
        os.path.join(OUTDIR, "label_names.npy"),
        np.array([[el] for el in human_readable_cnames]),
    )

    train_wav_list = []
    train_label_list = []
    eval_wav_list = []
    eval_label_list = []
    for i in tqdm(range(0, len(data))):
        cur_label = human_readable_cnames.index(data.iloc[i][2])
        start_sec = int(data.iloc[i][1])
        train = data.iloc[i][3] == "train"
        cur_path = (
            f"{'train' if train else 'test'}/{classes[cur_label]}/"
            f"{data.iloc[i][0]}_{start_sec:06d}_{(start_sec+10):06d}.wav"
        )
        if not os.path.exists(os.path.join(DATADIR, cur_path)):
            continue
        if train:
            train_wav_list.append(cur_path)
            train_label_list.append(cur_label)
        else:
            eval_wav_list.append(cur_path)
            eval_label_list.append(cur_label)

    print(f"Found {len(train_wav_list) + len(eval_wav_list)}/{len(data)} items")
    np.save(os.path.join(OUTDIR, f"train_filelist.npy"), train_wav_list)
    np.save(os.path.join(OUTDIR, f"train_labels.npy"), train_label_list)
    np.save(os.path.join(OUTDIR, f"eval_filelist.npy"), eval_wav_list)
    np.save(os.path.join(OUTDIR, f"eval_labels.npy"), eval_label_list)


if __name__ == "__main__":
    main()
