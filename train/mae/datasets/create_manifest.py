#!/usr/bin/env python

import os
import sys
import glob

from tqdm import tqdm

if __name__ == "__main__":
    assert len(sys.argv) == 3
    datadir = sys.argv[1]
    manifest_path = sys.argv[2]

    extensions = {".jpg", ".jpeg", ".png"}
    print(
        f"Loading files from {datadir} with extensions {extensions} and writing to {manifest_path}"
    )

    n_img = 0

    def image_filepath_generator():
        folder = glob.glob(datadir + "/*")
        for f in folder:
            for ext in extensions:
                for filepath in glob.glob(f + "/*" + ext):
                    yield filepath

    with open(manifest_path, "w") as f:
        for image_filepath in tqdm(image_filepath_generator()):
            f.write(f"{image_filepath}\n")
            n_img += 1
    print(f"Found {n_img} images")
