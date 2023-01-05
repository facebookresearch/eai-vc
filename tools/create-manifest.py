import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str)
    parser.add_argument("-p", "--pattern", type=str, default="**/*.jpg")
    args = parser.parse_args()

    root = Path(args.root)

    print("searching for files in {}".format(root / args.pattern))
    files = sorted(root.glob(args.pattern))
    folders = set(Path(path).parent for path in files)
    print("found {:,} files in {:,} folders".format(len(files), len(folders)))

    fname = "manifest.txt"
    with open(fname, "w") as f:
        for path in files:
            f.write(str(path) + "\n")
    print("wrote manifest file to '{}' in the current directory".format(fname))


if __name__ == "__main__":
    main()
