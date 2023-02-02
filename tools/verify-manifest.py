import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=str)
    args = parser.parse_args()

    path = Path(args.manifest)
    assert path.exists() and path.suffix == ".txt"

    # load manifest
    print(f"loading {path}")
    with path.open("r") as f:
        files = [p.strip() for p in f.readlines()]

    folders = set([Path(p).parent for p in files])
    filenames = [Path(p).name for p in files]

    print("found {:,} files and {:,} folders".format(len(filenames), len(folders)))

    # REPEAT_FACTOR
    ratio = len(filenames) / len(folders)
    print()
    print("REPEAT_FACTOR={}  # ({:0.2f})".format(int(np.round(ratio)), ratio))

    # check filenames
    lengths = [len(fname) for fname in filenames]
    if min(lengths) != max(lengths):
        print()
        print("*" * 80)
        print(
            "Warning: the filenames are not equal length, so they might not be sorted\n"
            "correctly. Please check below."
        )
        print("*" * 80)

    # verify
    print("\nInfo: printing 5 random segments of the manifest to verify:")
    for i in range(5):
        idx = np.random.randint(len(files) - 5)
        print("\nsegment {}".format(i))
        for j in range(5):
            print(files[idx + j])


if __name__ == "__main__":
    main()
