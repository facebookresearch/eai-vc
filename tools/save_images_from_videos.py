# load videos from path and save images to path
import argparse
import glob
import os
import pickle
import tqdm
import subprocess
import json
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np


def with_ffprobe(filename, ignore_secs):
    """Get video length, fps and resolution using ffprobe."""
    try:
        result = subprocess.check_output(
            f'ffprobe -v quiet -show_streams -select_streams v:0 -of json "{filename}"',
            shell=True,
        ).decode()
    except subprocess.CalledProcessError as e:
        print("Error with video: ", filename)
        return 0, 0, (0, 0), 0
    fields = json.loads(result)["streams"][0]

    duration = float(fields["duration"])
    if duration < ignore_secs * 2:
        return 0, 0, (0, 0), 0
    else:
        duration -= ignore_secs * 2

    fps = eval(fields["r_frame_rate"])
    resolution = (fields["width"], fields["height"])
    bitrate = int(fields["bit_rate"])

    return duration, fps, resolution, bitrate


def get_video_stats(file_list, saving_fps, ignore_secs, image_save_path, store_images):
    """Get video stats and save images."""

    # get stats for each video including video name, video length, fps, resolution and number of frames
    video_stats = {}
    # tqdm for progress bar
    for i in tqdm.tqdm(range(len(file_list))):
        file_path = file_list[i]
        video_name = os.path.basename(file_path)
        video_id = video_name.split(".")[0]

        # get video length, fps and resolution
        video_length, fps, resolution, bitrate = with_ffprobe(file_path, ignore_secs)
        if video_length < 0.1:  # or bitrate < 310000:
            print(
                "Skipping video: ",
                video_id,
                " due to low bitrate or short length: ",
                bitrate,
                video_length,
            )
            continue

        # get number of frames
        num_frames = int(video_length * fps)
        if video_length < 1 / saving_fps:
            # use twice the fps if video is too short
            current_saving_fps = 2 / video_length
        else:
            current_saving_fps = saving_fps
        # get number of images
        num_images = int(video_length * current_saving_fps)
        # fps for saving images

        # save stats
        video_stats[video_id] = {
            "video_name": video_name,
            "video_length": video_length,
            "fps": fps,
            "resolution": resolution,
            "num_frames": num_frames,
            "num_images": num_images,
            "bitrate": bitrate,
        }

        if store_images:
            # create folder for images
            if not os.path.exists(image_save_path + "/" + video_id):
                os.system("mkdir " + image_save_path + "/" + video_id)

            # save images from video at current_saving_fps while ignoring the first 15 and last 15 seconds without verbose
            os.system(
                "ffmpeg -i "
                + file_path
                + " -vf fps="
                + str(current_saving_fps)
                + " -qscale:v 1 -loglevel quiet -ss "
                + str(ignore_secs)
                + " -t "
                + str(video_length - 2 * ignore_secs)
                + " "
                + image_save_path
                + "/"
                + video_id
                + "/%06d.jpg"
            )

    return video_stats


def print_stats(video_stats, image_save_path):
    """Print stats for video length, fps, resolution and number of frames"""
    video_length = []
    fps = []
    resolution = []
    num_frames = []
    num_images = []
    bitrate = []
    bad_videos = 0
    for video_id in video_stats:
        if video_stats[video_id]["num_images"] == 0:
            bad_videos += 1
            continue
        video_length.append(video_stats[video_id]["video_length"])
        fps.append(video_stats[video_id]["fps"])
        resolution.append(video_stats[video_id]["resolution"])
        num_frames.append(video_stats[video_id]["num_frames"])
        num_images.append(video_stats[video_id]["num_images"])
        bitrate.append(video_stats[video_id]["bitrate"])

    if not os.path.exists(image_save_path):
        os.system("mkdir " + image_save_path)
    print("Total number of images: ", sum(num_frames))
    print("Total number of images saved: ", sum(num_images))
    print("Total duration of videos in hours: ", sum(video_length) / 3600)
    print("Total number of videos: ", len(video_stats))
    print("Number of bad videos: ", bad_videos)
    print(
        "Percentage of videos with bitrate less than 310 kb/s: {} %".format(
            len([x for x in bitrate if x < 310000]) / len(bitrate) * 100
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Extract images from videos. 
                    The folder structure will be preserved. The images will be saved in the format: video_id/frame_id.jpg. 
                    The frame_id will be 6 digits long. The images will be saved at the fps specified by the --saving_fps 
                    argument. The images will be saved starting from the --ignore_secs seconds of the video. 
                    The images will be saved in the folder specified by the --image_save_path argument.
                    The video stats will be saved in the file specified by the --stats_file argument."""
    )
    parser.add_argument(
        "--path",
        type=str,
        default="/checkpoint/karmeshyadav/HowTo100M/videos",
        help="path to the folder with videos",
    )
    parser.add_argument(
        "--image_save_path",
        type=str,
        default="/checkpoint/maksymets/eaif/datasets/HT100M",
        help="path to the folder where images will be saved",
    )
    parser.add_argument(
        "--saving_fps", type=float, default=1 / 20, help="fps for saving images"
    )  # 1/3 for Kinetics Dataset
    parser.add_argument(
        "--ignore_secs",
        type=float,
        default=15,
        help="number of seconds to ignore at the beginning and at the end of the video",
    )  # 0 secs for Kinetics Dataset
    parser.add_argument(
        "--store_images",
        default=True,
        action="store_true",
        help="whether to store images or not",
    )
    parser.add_argument(
        "--dont_store_images",
        dest="store_images",
        action="store_false",
        help="whether to store images or not",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=20,
        help="number of processes for multiprocessing",
    )
    args = parser.parse_args()

    print("Path to the folder with videos: ", args.path)

    # # get list of all videos in the folder
    file_list = glob.glob(args.path + "/*.mp4")

    print("Number of videos: ", len(file_list))

    # file_list = file_list[:100]

    if not os.path.exists(args.image_save_path):
        os.system("mkdir " + args.image_save_path)

    # run get_video_stats with multiprocessing
    async_results = {}
    # split file_list into num_processes chunks and get video stats for each chunk asynchronously
    with mp.Pool(processes=args.num_processes) as pool:
        for i in range(args.num_processes):
            # get video stats for each chunk
            chunk = file_list[i :: args.num_processes]
            async_results[i] = pool.apply_async(
                get_video_stats,
                args=(
                    chunk,
                    args.saving_fps,
                    args.ignore_secs,
                    args.image_save_path,
                    args.store_images,
                ),
            )
        pool.close()
        pool.join()

    # merge video stats from all chunks
    video_stats = {}
    for i in range(args.num_processes):
        video_stats.update(async_results[i].get())

    with open(args.image_save_path + "/video_stats_2.pickle", "wb") as f:
        pickle.dump(video_stats, f)

    print_stats(video_stats, args.image_save_path)
