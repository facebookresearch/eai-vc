import os
import os.path as osp

import cv2
import numpy as np


def save_mp4(frames, vid_dir: str, name: str, fps: int = 60, should_print=True):
    """
    :param name: The name WITHOUT the ".mp4" extension.
    """
    frames = np.array(frames)
    if len(frames[0].shape) == 4:
        new_frames = frames[0]
        for i in range(len(frames) - 1):
            new_frames = np.concatenate([new_frames, frames[i + 1]])
        frames = new_frames

    if not osp.exists(vid_dir):
        os.makedirs(vid_dir)

    vid_file = osp.join(vid_dir, name + ".mp4")
    if osp.exists(vid_file):
        os.remove(vid_file)

    w, h = frames[0].shape[:-1]
    videodims = (h, w)
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video = cv2.VideoWriter(vid_file, fourcc, fps, videodims)
    for frame in frames:
        frame = frame[..., 0:3][..., ::-1]
        video.write(frame)
    video.release()
    if should_print:
        print(f"Rendered to {vid_file}")
