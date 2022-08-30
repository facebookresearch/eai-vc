import sys
import os
import datetime
import re
import numpy as np
import torch
import pandas as pd
from termcolor import colored
from omegaconf import OmegaConf


CONSOLE_FORMAT = [
    ("episode", "E", "int"),
    ("env_step", "S", "int"),
    ("episode_reward", "R", "float"),
    ("total_time", "T", "time"),
]
OFFLINE_CONSOLE_FORMAT = [
    ("iteration", "I", "int"),
    ("reward", "R", "float"),
    ("total_time", "T", "time"),
]
AGENT_METRICS = [
    "consistency_loss",
    "reward_loss",
    "value_loss",
    "total_loss",
    "weighted_loss",
    "pi_loss",
    "grad_norm",
]

CAT_TO_COLOR = {
    "train": "blue",
    "eval": "green",
    "offline": "yellow",
}


def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def print_run(cfg):
    """Pretty-printing of run information. Call at start of training."""
    prefix, color, attrs = "  ", "green", ["bold"]

    def limstr(s, maxlen=32):
        return str(s[:maxlen]) + "..." if len(str(s)) > maxlen else s

    def pprint(k, v):
        print(
            prefix + colored(f'{k.capitalize()+":":<16}', color, attrs=attrs), limstr(v)
        )

    kvs = [
        ("task", cfg.task_title),
        ("train steps", f"{int(cfg.train_steps*cfg.action_repeat):,}"),
        ("observations", "x".join([str(s) for s in cfg.obs_shape])),
        ("actions", cfg.action_dim),
        ("experiment", cfg.exp_name),
    ]
    w = np.max([len(limstr(str(kv[1]))) for kv in kvs]) + 21
    div = "-" * w
    print(div)
    for k, v in kvs:
        pprint(k, v)
    print(div)


def cfg_to_group(cfg, return_list=False):
    """Return a wandb-safe group name for logging. Optionally returns group name as list."""
    lst = [cfg.task, cfg.modality, re.sub("[^0-9a-zA-Z]+", "-", cfg.exp_name)]
    return lst if return_list else "-".join(lst)


class VideoRecorder:
    """Utility class for logging evaluation videos."""

    def __init__(self, root_dir, wandb, render_size=384, fps=15):
        self.save_dir = (root_dir / "eval_video") if root_dir else None
        self._wandb = wandb
        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.enabled = False

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir and self._wandb and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            frame = env.render(
                mode="rgb_array",
                height=self.render_size,
                width=self.render_size,
                camera_id=0,
            )
            if frame is not None:
                self.frames.append(frame)

    def save(self, step, key="videos/eval_video"):
        if self.enabled and len(self.frames) > 0:
            frames = np.stack(self.frames).transpose(0, 3, 1, 2)
            self._wandb.log(
                {key: self._wandb.Video(frames, fps=self.fps, format="mp4")}, step=step
            )


class Logger(object):
    """Primary logger object. Logs either locally or using wandb."""

    def __init__(self, log_dir, cfg):
        self._log_dir = make_dir(log_dir)
        self._model_dir = make_dir(self._log_dir / "models")
        self._save_model = cfg.save_model
        self._group = cfg_to_group(cfg)
        self._seed = cfg.seed
        self._eval = []
        print_run(cfg)
        project, entity = cfg.get("wandb_project", "none"), cfg.get(
            "wandb_entity", "none"
        )
        run_offline = project == "none" or entity == "none"
        if run_offline:
            print(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
            self._wandb = None
        else:
            try:
                os.environ["WANDB_SILENT"] = "true"
                import wandb

                wandb.init(
                    project=project,
                    entity=entity,
                    name=str(cfg.seed),
                    group=self._group,
                    tags=cfg_to_group(cfg, return_list=True) + [f"seed:{cfg.seed}"],
                    dir=self._log_dir,
                    config=OmegaConf.to_container(cfg, resolve=True),
                )
                print(
                    colored("Logs will be synced with wandb.", "blue", attrs=["bold"])
                )
                self._wandb = wandb
            except:
                print(
                    colored(
                        "Warning: failed to init wandb. Logs will be saved locally.",
                        "yellow",
                        attrs=["bold"],
                    )
                )
                self._wandb = None
        self._video = (
            VideoRecorder(log_dir, self._wandb)
            if self._wandb and cfg.save_video
            else None
        )

    @property
    def video(self):
        return self._video

    @property
    def model_dir(self):
        return self._model_dir

    def save_model(self, agent, identifier, metadata={}):
        if self._save_model:
            fp = self._model_dir / f"{str(identifier)}.pt"
            agent.save(fp, metadata)
            if self._wandb:
                artifact = self._wandb.Artifact(
                    self._group + "-" + str(self._seed) + "-" + str(identifier),
                    type="model",
                )
                artifact.add_file(fp)
                self._wandb.log_artifact(artifact)

    def finish(self):
        if self._wandb:
            self._wandb.finish()

    def _format(self, key, value, ty):
        if ty == "int":
            return f'{colored(key+":", "grey")} {int(value):,}'
        elif ty == "float":
            return f'{colored(key+":", "grey")} {value:.01f}'
        elif ty == "time":
            value = str(datetime.timedelta(seconds=int(value)))
            return f'{colored(key+":", "grey")} {value}'
        else:
            raise f"invalid log format type: {ty}"

    def _print(self, d, category):
        _format = OFFLINE_CONSOLE_FORMAT if category == "offline" else CONSOLE_FORMAT
        category = colored(category, CAT_TO_COLOR[category])
        pieces = [f" {category:<14}"]
        for k, disp_k, ty in _format:
            pieces.append(f"{self._format(disp_k, d.get(k, 0), ty):<26}")
        print("   ".join(pieces))

    def log(self, d, category="train"):
        assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
        if self._wandb is not None:
            if category in {"train", "eval"}:
                xkey = "env_step"
            elif category == "offline":
                xkey = "iteration"
            for k, v in d.items():
                self._wandb.log({category + "/" + k: v}, step=d[xkey])
        if category == "eval":
            keys = ["env_step", "episode_reward"]
            self._eval.append(np.array([d[keys[0]], d[keys[1]]]))
            pd.DataFrame(np.array(self._eval)).to_csv(
                self._log_dir / "eval.log", header=keys, index=None
            )
        self._print(d, category)
