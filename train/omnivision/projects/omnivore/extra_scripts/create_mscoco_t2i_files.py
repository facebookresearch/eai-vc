import json
import os

import numpy as np
from tqdm import tqdm


DATADIR = "/datasets01/large_experiments/cmd/coco/images/"
OUTDIR = "/checkpoint/aelnouby/datasets/mscoco"
KARPATHY_SPLIT = "/checkpoint/aelnouby/datasets/mscoco/dataset_coco.json"


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    with open(KARPATHY_SPLIT, "r") as f:
        karpathy_split = json.load(f)["images"]

    test_split = [sample for sample in karpathy_split if sample["split"] == "test"]

    fpaths = []
    captions = []
    text2img_mapping = []
    extra_captions_count = 0
    assert len(test_split) == 5000
    for idx, entry in enumerate(tqdm(test_split)):
        entry_captions = [e["raw"] for e in entry["sentences"]]
        if len(entry_captions) > 5:
            extra_captions_count += 1
        entry_captions = entry_captions[:5]
        fpath = os.path.join(DATADIR, entry["filepath"], entry["filename"])
        assert os.path.exists(fpath), f"{fpath} is not found"

        fpaths.append(fpath)
        captions.extend(entry_captions)
        text2img_mapping.extend([idx] * len(entry_captions))

    np.save(os.path.join(OUTDIR, f"filelist.npy"), fpaths)
    np.save(os.path.join(OUTDIR, f"captions.npy"), captions)
    np.save(os.path.join(OUTDIR, f"text2img_mapping.npy"), text2img_mapping)
    _, counts = np.unique(text2img_mapping, return_counts=True)
    assert np.all(counts == counts[0])
    print("Done...")
    print(f"There are {extra_captions_count} entries with one extra caption.")


if __name__ == "__main__":
    main()
