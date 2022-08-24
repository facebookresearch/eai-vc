import glob
import os

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

import models_mae

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def show_image(image, title=""):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis("off")
    return


def prepare_model(chkpt_dir, arch="mae_vit_large_patch16"):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location="cpu")
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    print(msg)
    return model


def run_one_image(img, model, image_name):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum("nhwc->nchw", x)

    # run MAE
    loss, y, mask = model(x.float(), x.float(), mask_ratio=0.75)
    print("Loss on {}: {}".format(image_name, loss.item()))
    y = model.unpatchify(y)
    y = torch.einsum("nchw->nhwc", y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(
        1, 1, model.patch_embed.patch_size[0] ** 2 * 3
    )  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum("nchw->nhwc", mask).detach().cpu()

    x = torch.einsum("nchw->nhwc", x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams["figure.figsize"] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")

    folder_name = "examples/output/"
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    plt.savefig("{}/{}".format(folder_name, image_name))


if __name__ == "__main__":
    chkpt_dir = "../data/ddppo-models/mae_small_01.pth"
    model_mae = prepare_model(chkpt_dir, "mae_vit_small_patch16")
    print("Model loaded.")

    img_paths = glob.glob("examples/*.*")
    assert len(img_paths) > 0
    # load images
    imgs = []

    for path in img_paths:
        img = Image.open(path)

        img = img.resize((224, 224))
        img = np.array(img) / 255.0

        assert img.shape == (224, 224, 3)

        # normalize by ImageNet mean and std
        img = img - imagenet_mean
        img = img / imagenet_std

        # make random mask reproducible (comment out to make it change)
        torch.manual_seed(2)
        print("MAE with pixel reconstruction:")
        run_one_image(img, model_mae, path.split("/")[-1])
