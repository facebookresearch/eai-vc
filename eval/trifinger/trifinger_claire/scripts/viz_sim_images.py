import numpy as np
import argparse
import sys
import os
import cv2
import imageio

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))

import utils.data_utils as d_utils

def main(args):
    
    data = np.load(args.file_path, allow_pickle=True)["data"]
    print(len(data))

    demo_name = os.path.splitext(os.path.split(args.file_path)[1])[0]
    demo_dir = os.path.split(args.file_path)[0]
    out_dir = os.path.join(demo_dir, "sim_viz")
    if not os.path.exists(out_dir): os.makedirs(out_dir, exist_ok=False)

    traj_original = d_utils.get_traj_dict_from_obs_list(data)
    traj = d_utils.downsample_traj_dict(traj_original, new_time_step=0.2)
    frames = []
    for i, img in enumerate(traj["image_60"]):
        #img = d_utils.resize_img(img).detach().numpy().transpose(1,2,0) * 255.
        frames.append(img.astype(np.uint8))
    imageio.mimsave(os.path.join(out_dir, f'{demo_name}.gif'), frames)

    if args.video:
        out_dir = os.path.join(demo_dir, "sim_viz", demo_name)
        if not os.path.exists(out_dir): os.makedirs(out_dir, exist_ok=False)

        for cam_name in ["camera60", "camera180", "camera300"]:
            img_list = []
            for i in range(len(data)):
                img = data[i]["camera_observation"][cam_name]["image"]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                height, width, layers = img.shape
                img_list.append(img)

            size = (height, width)

            out_name = f"{cam_name}.mp4"
            out_path = os.path.join(out_dir, out_name)

            out_file = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 1000, size)

            for i in range(len(img_list)):
                out_file.write(img_list[i])
            out_file.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", default=None, help="""Filepath of trajectory to load""")
    parser.add_argument("--video", "-v", action="store_true", help="Save .mp4 video")
    args = parser.parse_args()
    main(args)


