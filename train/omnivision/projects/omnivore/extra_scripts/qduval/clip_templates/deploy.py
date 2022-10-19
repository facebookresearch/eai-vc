import argparse
import os

import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output", type=str, default="/checkpoint/omniscale_oss/clip_templates"
    )
    args = parser.parse_args()
    out_folder = os.path.abspath(os.path.expanduser(args.output))

    for file_path in os.listdir(""):
        file_name, file_ext = os.path.splitext(file_path)
        if file_ext == ".txt":
            with open(file_path, "r") as f:
                lines = [line.strip() for line in f if line]
            out_file_path = os.path.join(out_folder, f"{file_name}.npy")
            print(file_path, "to:", out_file_path)
            np.save(out_file_path, lines)
