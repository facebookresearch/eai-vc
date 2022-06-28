import numpy as np
import argparse
import os
import os.path
import cv2

def main(file_path):
    data = np.load(file_path, allow_pickle=True)["data"]
    print(len(data))

    #print(data[-1]["achieved_goal"]["position_error"]) # final position error

    for cam_name in ["camera60", "camera180", "camera300"]:
        img_list = []
        for i in range(len(data)):
            img = data[i]["camera_observation"][cam_name]["image"]
            height, width, layers = img.shape
            img_list.append(img)
            
        size = (height, width)

        demo_name = os.path.splitext(os.path.split(file_path)[1])[0]
        demo_dir = os.path.split(file_path)[0]
        out_dir = os.path.join(demo_dir, "viz", demo_name)
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        out_name = f"{cam_name}.mp4"
        out_path = os.path.join(out_dir, out_name)

        out_file = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 1000, size)
        
        for i in range(len(img_list)):
            out_file.write(img_list[i])
        out_file.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", default=None, help="""Filepath of trajectory to load""")
    args = parser.parse_args()
    main(args.file_path)


