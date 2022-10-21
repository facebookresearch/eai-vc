import os
import subprocess

import numpy as np
import pandas
from tqdm import tqdm

# Based on https://www.robots.ox.ac.uk/~vgg/research/audio-retrieval/
# Note that test/val sets have 5 captions to each audio, training has 1 caption for each audio
DATADIR = "/datasets01/audioset/042319/"
ACTUAL_DATADIR = os.path.join(DATADIR, "data/")
DATATYPEDIR = "audio"
DATATYPEEXT = ".flac"
OUTDIR = "/checkpoint/rgirdhar/Work/FB/2021/003_JointImVid/Datasets/AudioCapsRetrieval/"
SOURCE = "https://www.robots.ox.ac.uk/~vgg/research/audio-retrieval/resources/benchmark-files/AudioCaps_retrieval_dataset.tar.gz"


def download_files():
    if os.path.exists(f"{OUTDIR}/AudioCaps_retrieval_dataset.tar.gz"):
        # Already downloaded and extracted
        return
    subprocess.run(
        f"cd {OUTDIR}; wget {SOURCE}; tar xf AudioCaps_retrieval_dataset.tar.gz",
        shell=True,
    )


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    download_files()
    for split in ["val", "test", "train"]:
        csv_file = os.path.join(
            OUTDIR,
            f"AudioCaps_retrieval_dataset/retrieval_{split}.csv",
        )
        data = pandas.read_csv(csv_file)
        grouped = data.groupby(["youtube_id", "start_time"])
        fpaths = []
        captions = []
        cap2audio = []
        for name, group in tqdm(grouped, desc=split, total=len(grouped)):
            this_captions = group["caption"].values.tolist()
            fpath = f"{name[0]}_{name[1] * 1000}_{(name[1] + 10) * 1000}{DATATYPEEXT}"
            final_fpath = None
            for folder in [
                "balanced_train_segments",
                "eval_segments",
                "unbalanced_train_segments",
            ]:
                if os.path.exists(
                    os.path.join(ACTUAL_DATADIR, folder, DATATYPEDIR, fpath)
                ):
                    final_fpath = os.path.join(folder, DATATYPEDIR, fpath)
                    break
            if final_fpath is None:
                continue
            fpaths.append(final_fpath)
            captions += this_captions
            idx_of_vid = len(fpaths) - 1
            cap2audio += [idx_of_vid] * len(this_captions)
        np.save(os.path.join(OUTDIR, f"{split}_filelist.npy"), fpaths)
        np.save(os.path.join(OUTDIR, f"{split}_captions.npy"), captions)
        np.save(os.path.join(OUTDIR, f"{split}_captions2audio.npy"), cap2audio)
        perc_data_found = (len(fpaths) / len(grouped)) * 100.0
        print(f"Done for {split}. Found {perc_data_found:0.02f}% ")


if __name__ == "__main__":
    main()
