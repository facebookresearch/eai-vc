import json

import torch.nn as nn
from iopath.common.file_io import g_pathmgr


class ImagenetIdtoIdxAirstore(nn.Module):
    """
    To be used with the Airstore Imagenet experiments as the
    airstore table only has image names.
    """

    def __init__(self, imagenet_ids_to_idx_path):
        super().__init__()
        with g_pathmgr.open(imagenet_ids_to_idx_path, "rb") as f:
            self.imagenet_ids_to_idx = json.load(f)

    def forward(self, img_path):
        """
        Args:
            Path to the image file.
        Returns:
            Class idx of image.
        """
        # Image path of form "tree/imagenet/train/n02111129/n02111129_3736.JPEG"
        class_idx = int(self.imagenet_ids_to_idx[img_path.split("/")[3]])
        return class_idx
