from typing import Any, Callable

import torch


class ImageToSingleFrameVideo(Callable):
    def __call__(self, image_tensor):
        """Converts (N x) C x H X W image to a (N x) C x T x H x W (T = 1) video frame."""
        if image_tensor.ndim == 3:
            return image_tensor[:, None, ...]
        assert image_tensor.ndim == 4
        return image_tensor[:, :, None, ...]


class RepeatedPadIm2VideoSingleImage(torch.nn.Module):
    def __init__(self, ntimes, time_dim=1):
        super().__init__()
        assert ntimes > 0
        self.ntimes = ntimes
        self.time_dim = time_dim

    def forward(self, x):
        # C x H x W -> C x T x H x W
        if x.ndim == 5:
            return x
        x = x.unsqueeze(self.time_dim)
        new_shape = [1] * len(x.shape)
        new_shape[self.time_dim] = self.ntimes
        x = x.repeat(new_shape)
        return x
