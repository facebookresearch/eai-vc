import os

import numpy as np
import pandas
from tqdm import tqdm

# This is the 1k-A split of the data
DATADIR = "/fsx-omnivore/rgirdhar/data/MSRVTT/videos/all"
OUTDIR = "/checkpoint/rgirdhar/Work/FB/2021/003_JointImVid/Datasets/MSR-VTT/1k-A/"
SOURCE = "https://raw.githubusercontent.com/antoine77340/MIL-NCE_HowTo100M/master/csv/msrvtt_test.csv"


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    data = pandas.read_csv(SOURCE)
    fpaths = []
    captions = []
    cap2audio = []
    for _, row in tqdm(data.iterrows(), total=len(data)):
        this_captions = [row.sentence]
        fpath = f"{DATADIR}/{row.video_id}.mp4"
        if not os.path.exists(fpath):
            continue
        fpaths.append(fpath)
        captions += this_captions
        idx_of_vid = len(fpaths) - 1
        cap2audio += [idx_of_vid] * len(this_captions)
    np.save(os.path.join(OUTDIR, f"filelist.npy"), fpaths)
    np.save(os.path.join(OUTDIR, f"captions.npy"), captions)
    np.save(os.path.join(OUTDIR, f"captions2audio.npy"), cap2audio)
    perc_data_found = (len(fpaths) / len(data)) * 100.0
    print(f"Done. Found {perc_data_found:0.02f}% ")


if __name__ == "__main__":
    main()
