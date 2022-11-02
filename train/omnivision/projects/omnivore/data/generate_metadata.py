import os
import glob
import numpy as np


def generate_path_list(root_path, extensions, savepath):
    root_path = os.path.abspath(root_path)

    result = []
    for extension in extensions:
        result.extend(glob.glob(f"{root_path}/**/*.{extension}", recursive=True))

    np_result = np.array(result)
    print(f"=== Saving length {len(np_result)} dataset: ===")
    print(np_result)

    savepath = savepath or os.path.join(root_path, "path_file_list.npy")
    print(f"=== Saving to {savepath} ===")
    np.save(savepath, np_result)
    print(f"Success.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate npy file containing a numpy array of all absolute paths of files with extension `ext` under `path`, saved in `savepath`."
    )
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--ext", type=str, nargs="+", required=True)
    parser.add_argument("--savepath", type=str, required=False)
    args = parser.parse_args()

    generate_path_list(args.path, args.ext, args.savepath)
