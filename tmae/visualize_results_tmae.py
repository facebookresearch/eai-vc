import glob
import os

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

import models_mae

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16', decoder_max_offset=16):
    # build model
    model = getattr(models_mae, arch)(decoder_max_offset=decoder_max_offset)
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def pre_forward(img):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    return x

def post_forward(x, y, mask, model):
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    return x, y, im_masked, im_paste

def run_one_example(imgs, idx, offset, model, img_paths):
    idx_1 = idx
    idx_2 = idx_1 + offset

    path_1 = img_paths[idx_1].split('/')[-1]
    path_2 = img_paths[idx_2].split('/')[-1]

    x1 = pre_forward(imgs[idx_1])
    x2 = pre_forward(imgs[idx_2])

    # run MAE
    loss, y, mask = model(x1.float(), x1.float(), x2.float(), x2.float(), offsets=torch.tensor(offset), mask_ratio1=0.75, mask_ratio2=0.95)
    print("Loss 1 on {}: {} \nLoss 2 on {}: {}".format(path_1, loss[0].item(), path_2, loss[1].item()))

    # post-process
    x1, y1, im_masked1, im_paste1 = post_forward(x1, y[0], mask[0], model)
    x2, y2, im_masked2, im_paste2 = post_forward(x2, y[1], mask[1], model)

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]
    plt.subplot(2, 4, 1)
    show_image(x1[0], "original")

    plt.subplot(2, 4, 2)
    show_image(im_masked1[0], "masked")

    plt.subplot(2, 4, 3)
    show_image(y1[0], "reconstruction")

    plt.subplot(2, 4, 4)
    show_image(im_paste1[0], "reconstruction + visible")

    plt.subplot(2, 4, 5)
    show_image(x2[0], "original")

    plt.subplot(2, 4, 6)
    show_image(im_masked2[0], "masked")

    plt.subplot(2, 4, 7)
    show_image(y2[0], "reconstruction")

    plt.subplot(2, 4, 8)
    show_image(im_paste2[0], "reconstruction + visible")

    folder_name = "examples/output/"
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    plt.savefig("{}/{}_{}.jpg".format(folder_name, path_1.split(".")[0], path_2.split(".")[0]))

if __name__ == '__main__':
    chkpt_dir = '../data/ddppo-models/tmae_small_offset_4_viz.pth'
    model_mae = prepare_model(chkpt_dir, 'mae_vit_small_patch16', decoder_max_offset=4)
    print('Model loaded.')

    img_paths = glob.glob('examples/hm3d_trajectory/*.jpg')
    assert len(img_paths) > 0
    # sort the images by name
    img_paths = sorted(img_paths)

    # load images
    imgs = []
    
    for path in img_paths:
        img = Image.open(path)

        img = img.resize((224, 224))
        img = np.array(img) / 255.

        assert img.shape == (224, 224, 3)

        # normalize by ImageNet mean and std
        img = img - imagenet_mean
        img = img / imagenet_std

        imgs.append(img)

    for offset in range(5):
        # make random mask reproducible (comment out to make it change)
        torch.manual_seed(2)
        print('MAE with pixel reconstruction:')
        run_one_example(imgs, 0, offset, model_mae, img_paths)



