# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import omegaconf
import hydra
import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from r3m import load_r3m


"""
adapted from https://github.com/facebookresearch/r3m/blob/eval/r3m/example.py
to work with demonstration data from Trifinger tasks
"""

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

r3m = load_r3m("resnet50")  # resnet18, resnet34
r3m.eval()
r3m.to(device)

## DEFINE PREPROCESSING
transforms = T.Compose(
    [T.Resize(256), T.CenterCrop(224), T.ToTensor()]
)  # ToTensor() divides by 255

file_path = "/Users/fmeier/projects/claire_ws/demos/difficulty-1/demo-0000.npz"
demo_data = np.load(file_path, allow_pickle=True)["data"]

image = demo_data[0]["camera_observation"]["camera60"]["image"]
preprocessed_image = transforms(Image.fromarray(image.astype(np.uint8))).reshape(
    -1, 3, 224, 224
)
plt.imshow(image)
plt.show()
plt.imshow(preprocessed_image[0].transpose(0, 2).transpose(0, 1))
plt.show()

with torch.no_grad():
    embedding = r3m(
        preprocessed_image * 255.0
    )  ## R3M expects image input to be [0-255]
print(embedding.shape)  # [1, 2048]
