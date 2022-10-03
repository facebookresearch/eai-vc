import itertools
from typing import Dict, Optional

import torch
from torchvision.io import VideoReader


class RealTorchvisionVideoReader:
    def __init__(self, path, device="cpu"):
        self.path = path
        self.video_name = path
        self.reader = VideoReader(path, stream="video", device=device)
        self.metadata = self.reader.get_metadata()
        self.duration = self.metadata["video"]["duration"][0]
        self.fps = self.metadata["video"]["fps"][0]

    def get_frames(self, start_idx, end_idx, end_strict=False):
        self.reader.seek()

    def get_clip(
        self, start_sec: float, end_sec: float
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Retrieves frames from the encoded video at the specified start and end times
        in seconds (the video always starts at 0 seconds).

        Args:
            start_sec (float): the clip start time in seconds
            end_sec (float): the clip end time in seconds
        Returns:
            clip_data:
                A dictionary mapping the entries at "video" and "audio" to a tensors.

                "video": A tensor of the clip's RGB frames with shape:
                (channel, time, height, width). The frames are of type torch.float32 and
                in the range [0 - 255].

                "audio": A tensor of the clip's audio samples with shape:
                (samples). The samples are of type torch.float32 and
                in the range [0 - 255].

            Returns None if no video or audio found within time range.

        """
        if start_sec > end_sec or start_sec > self.duration:
            raise RuntimeError(
                f"Incorrect time window for torchvision decoding for video: {self.video_name}."
            )

        frames = []
        # WARNING: This will return frames till at most end_sec, but can end sooner
        for frame in itertools.takewhile(
            lambda x: x["pts"] <= end_sec, self.reader.seek(start_sec)
        ):
            frames.append(frame["data"])

        clip = torch.stack(frames, dim=1)

        return {
            # this reader returns videos in uint8 format
            "video": clip.to(dtype=torch.float32),
        }
