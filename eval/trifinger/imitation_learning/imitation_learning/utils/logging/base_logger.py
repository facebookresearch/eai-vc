import datetime
import os
import os.path as osp
import string
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Set, Union

import numpy as np
import torch.nn as nn
from omegaconf import DictConfig

LoggerCfgType = Union[Dict[str, Any], DictConfig]


class Logger:
    def __init__(
        self,
        run_name: str,
        seed: int,
        log_dir: str,
        vid_dir: str,
        save_dir: str,
        smooth_len: int,
        full_cfg: LoggerCfgType,
        **kwargs,
    ):
        """
        :param run_name: If empty string then a run name will be auto generated.
        """
        self._create_run_name(run_name, seed)

        self.log_dir = log_dir
        if self.log_dir != "":
            self.log_dir = osp.join(self.log_dir, self.run_name)
            if not osp.exists(self.log_dir):
                os.makedirs(self.log_dir)

        self.vid_dir = vid_dir
        if self.vid_dir != "":
            self.vid_dir = osp.join(self.vid_dir, self.run_name)
            if not osp.exists(self.vid_dir):
                os.makedirs(self.vid_dir)

        self.save_dir = save_dir
        if self.save_dir != "":
            self.save_dir = osp.join(self.save_dir, self.run_name)
            if not osp.exists(self.save_dir):
                os.makedirs(self.save_dir)

        self._step_log_info = defaultdict(lambda: deque(maxlen=smooth_len))

        self.is_printing = True
        self.prev_steps = 0
        self.start = time.time()
        self._clear_keys: Set[str] = set()

    @property
    def save_path(self):
        return self.save_dir

    @property
    def vid_path(self):
        return self.vid_dir

    def disable_print(self):
        self.is_printing = False

    def collect_env_step_info(
        self, infos: List[Dict[str, Any]], step_info_keys: List[str]
    ) -> None:
        if step_info_keys:
            for key in step_info_keys:
                if key in infos.keys():
                    self._step_log_info[key].append(infos[key].cpu().numpy())

    def collect_infos(
        self, info: Dict[str, float], prefix: str = "", no_rolling_window: bool = False
    ) -> None:
        for k, v in info.items():
            self.collect_info(k, v, prefix, no_rolling_window)

    def collect_info(
        self, k: str, value: float, prefix: str = "", no_rolling_window: bool = False
    ) -> None:
        """
        :param no_rolling_window: If true, then only the most recent logged
            value will be displayed with a call to `self.interval_log`. This is for
            metrics that should not be averaged.
        """
        use_k = prefix + k
        if no_rolling_window:
            self._step_log_info[use_k].clear()
            self._clear_keys.add(use_k)
        self._step_log_info[use_k].append(value)

    def collect_info_list(self, k: str, values: List[float], prefix: str = "") -> None:
        """
        Collect a list of values for a key.
        """
        for v in values:
            self.collect_info(k, v)

    def _create_run_name(self, run_name, seed):
        if run_name == "":
            d = datetime.datetime.today()
            date_id = "%i%i" % (d.month, d.day)

            chars = list(
                string.ascii_uppercase + string.digits + string.ascii_lowercase
            )
            rnd_id = np.random.RandomState().choice(chars, 6)
            rnd_id = "".join(rnd_id)

            self.run_name = f"{date_id}-{seed}-{rnd_id}"
        else:
            self.run_name = run_name
        print(f"Assigning full prefix {self.run_name}")

    def log_vals(self, key_vals, step_count):
        """
        Log key value pairs to whatever interface.
        """

    def collect_img(self, k: str, img_path: str, prefix: str = ""):
        """
        Log an image
        :param img_path: Full path to the image.
        """

    def watch_model(self, model: nn.Module):
        """
        :param model: the set of parameters to watch
        """

    def interval_log(self, update_count: int, processed_env_steps: int) -> None:
        """
        Printed FPS is all inclusive of updates, evaluations, logging and everything.
        This is NOT the environment FPS.
        :param update_count: The number of updates.
        :param processed_env_steps: The number of environment samples processed.
        """
        end = time.time()

        fps = int((processed_env_steps - self.prev_steps) / (end - self.start))
        self.prev_steps = processed_env_steps
        num_eps = len(self._step_log_info.get("episode.reward", []))

        log_dat = {}
        for k, v in self._step_log_info.items():
            if isinstance(v, deque):
                log_dat[k] = np.mean(v)
            else:
                log_dat[k] = v

        for k in self._clear_keys:
            del self._step_log_info[k]
        self._clear_keys.clear()

        if self.is_printing:
            print("")
            print(f"Updates {update_count}, Steps {processed_env_steps}, FPS {fps}")
            print(f"Over the last {num_eps} episodes:")

            # Print log values from the updater if requested.
            for k, v in log_dat.items():
                print(f"    - {k}: {v}")
            print("", flush=True)

        # Log all values
        log_dat["fps"] = fps
        self.log_vals(log_dat, processed_env_steps)
        self.start = end

    def close(self):
        pass
