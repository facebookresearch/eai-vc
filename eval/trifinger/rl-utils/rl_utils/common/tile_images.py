import numpy as np


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.

    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)

    # NOTE: Added to support renderings that return images series.
    if len(img_nhwc.shape) == 1:
        # Pad other sequences so they are as long.
        dims = [img_nhwc[i].shape[0] for i in range(img_nhwc.shape[0])]
        max_dim = max(dims)
        for i in range(img_nhwc.shape[0]):
            if img_nhwc[i].shape[0] < max_dim:
                img_nhwc[i] = np.pad(
                    img_nhwc[i], ((0, 1), (0, 0), (0, 0), (0, 0)), "edge"
                )
        img_nhwc = np.array(list(img_nhwc))
        return tile_images(img_nhwc)

    if len(img_nhwc.shape) == 5:
        return np.array([tile_images(img_nhwc[:, i]) for i in range(img_nhwc.shape[1])])

    N, h, w, c = img_nhwc.shape

    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    # add boundary to distinguish environments
    img_HWhwc[:, :, :, -1, :] = 0
    img_HWhwc[:, :, -1, :, :] = 0
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    return img_HhWwc.reshape(H * h, W * w, c)
