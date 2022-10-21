import os

import numpy as np
import pandas
from tqdm import tqdm

# Downloaded the data and removed the spaces from the audio file names using
# https://unix.stackexchange.com/a/405089
DATADIR = "/fsx-omnivore/rgirdhar/data/Clotho/{}/"
OUTDIR = "/checkpoint/rgirdhar/Work/FB/2021/003_JointImVid/Datasets/Clotho/"
SOURCE = "https://zenodo.org/record/3490684/files/clotho_captions_{}.csv"


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    for split in ["evaluation", "development"]:
        data = pandas.read_csv(SOURCE.format(split))
        fpaths = []
        captions = []
        cap2audio = []
        for _, row in tqdm(data.iterrows(), desc=split, total=len(data)):
            this_captions = [
                row.caption_1,
                row.caption_2,
                row.caption_3,
                row.caption_4,
                row.caption_5,
            ]
            fpath = f"{DATADIR.format(split)}/{row.file_name.replace(' ', '_')}"
            if not os.path.exists(fpath):
                continue
            fpaths.append(fpath)
            captions += this_captions
            idx_of_vid = len(fpaths) - 1
            cap2audio += [idx_of_vid] * len(this_captions)
        np.save(os.path.join(OUTDIR, f"{split}_filelist.npy"), fpaths)
        np.save(os.path.join(OUTDIR, f"{split}_captions.npy"), captions)
        np.save(os.path.join(OUTDIR, f"{split}_captions2audio.npy"), cap2audio)
        perc_data_found = (len(fpaths) / len(data)) * 100.0
        print(f"Done for {split}. Found {perc_data_found:0.02f}% ")


if __name__ == "__main__":
    main()
