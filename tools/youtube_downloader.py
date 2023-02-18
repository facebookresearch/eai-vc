# download videos from youtube if we have bad quality videos locally
import argparse
import glob
import os
import tqdm
import subprocess
import json
import multiprocessing as mp
import yt_dlp


def with_ffprobe(filename):
    """Get video length, fps and resolution using ffprobe."""

    try:
        result = subprocess.check_output(
            f'ffprobe -v quiet -show_streams -select_streams v:0 -of json "{filename}"',
            shell=True,
        ).decode()
    except subprocess.CalledProcessError as e:
        return 0, 0, (0, 0), 0
    fields = json.loads(result)["streams"][0]

    duration = float(fields["duration"])
    if duration < 30:
        return 0, 0, (0, 0), 0
    else:
        duration -= 30

    fps = eval(fields["r_frame_rate"])
    resolution = (fields["width"], fields["height"])
    bitrate = int(fields["bit_rate"])

    return duration, fps, resolution, bitrate


resolution_preference_order = ["480p", "360p", "720p", "1080p"]


def log_failure_cases(video_save_path, video_id, e):
    """Log failure cases."""

    # save a file video_id.txt in video_save_path with the error message
    with open(video_save_path + "/" + video_id + ".txt", "w") as f:
        f.write(e)


def get_video_stats(file_list, video_save_path):
    """Get video stats and save images. Resolution preference order is a list of resolutions in order of preference.
    The function will try to download the video in the first resolution in the list.
    If the video is not available in that resolution, it will try the next resolution in the list.
    If the video is not available in any of the resolutions in the list, it will log the failure case and move on to the next video.
    If the video bitrate is less than 310000, it will log the failure case and move on to the next video.
    """

    for i in tqdm.tqdm(range(len(file_list))):
        file_path = file_list[i]
        video_name = os.path.basename(file_path)
        video_id = video_name.split(".")[0]

        # get video length, fps and resolution
        video_length, fps, resolution, bitrate = with_ffprobe(file_path)
        if video_length < 0.1:
            continue

        if bitrate < 310000:
            new_file_path = video_save_path + "/" + video_id + ".mp4"
            if os.path.exists(new_file_path) or os.path.exists(
                new_file_path.replace("mp4", "txt")
            ):
                continue
            # download better quality video from youtube
            video_downloaded = False
            for res in resolution_preference_order:
                try:
                    # use youtube-dl to download video
                    ydl_opts = {
                        "format": "bestvideo[ext=mp4][height<={}]".format(res[:-1]),
                        "outtmpl": new_file_path,
                        "quiet": True,
                        "no_warnings": True,
                        "nocheckcertificate": True,
                        "http-chunk-size": 1048000,
                    }
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download(["https://www.youtube.com/watch?v=" + video_id])
                        video_downloaded = True
                        break
                except Exception as e:
                    if "Network" in str(e):
                        continue
                    log_failure_cases(video_save_path, video_id, str(e))
                    continue
            if not video_downloaded:
                log_failure_cases(video_save_path, video_id, "video not downloaded")
                continue
        else:
            continue


if __name__ == "__main__":
    # argparse
    # Add help description to the script
    parser = argparse.ArgumentParser(
        description="""Donwloads videos from youtube.      The scrip will try to download the video in the first resolution in the list. 
    If the video is not available in that resolution, it will try the next resolution in the list. 
    If the video is not available in any of the resolutions in the list, it will log the failure case and move on to the next video. 
    If the video bitrate is less than 310000, it will log the failure case and move on to the next video."""
    )
    parser.add_argument(
        "--path",
        type=str,
        default="/datasets01/HowTo100M/022520/videos",
        help="path to the folder with videos",
    )
    parser.add_argument(
        "--video_save_path",
        type=str,
        default="/checkpoint/karmeshyadav/HowTo100M/videos",
        help="path to the folder where videos will be saved",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=20,
        help="number of processes for multiprocessing",
    )
    parser.add_argument(
        "--download_subset",
        type=int,
        default=0,
        help="which subset of video to download",
    )
    args = parser.parse_args()

    print("Path to the folder with videos: ", args.path)

    # # get list of all videos in the folder
    file_list = glob.glob(args.path + "/*.mp4")

    print("Number of videos: ", len(file_list))

    # divide list into 5 parts because we want to download from machines with different ip addresses
    print("Downloading subset: ", args.download_subset)
    file_list = file_list[args.download_subset :: 5]

    # split file_list into num_processes chunks and get video stats for each chunk asynchronously
    with mp.Pool(processes=args.num_processes) as pool:
        for i in range(args.num_processes):
            # get video stats for each chunk
            chunk = file_list[i :: args.num_processes]
            pool.apply_async(get_video_stats, args=(chunk, args.video_save_path))
        pool.close()
        pool.join()
