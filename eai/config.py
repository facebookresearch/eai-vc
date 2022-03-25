import warnings
from typing import List, Optional, Union

from habitat.config.default import _C as _HABITAT_CONFIG
from habitat.config.default import Config as CN
from habitat_baselines.config.default import _C as _BASE_CONFIG

CONFIG_FILE_SEPARATOR = ","

# -----------------------------------------------------------------------------
# TASK CONFIG
# -----------------------------------------------------------------------------

# fmt:off
_TASK_CONFIG = _HABITAT_CONFIG.clone()
_TASK_CONFIG.defrost()

_TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = 1000
_TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = 50000

_TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE = 0.25
_TASK_CONFIG.SIMULATOR.TURN_ANGLE = 30
_TASK_CONFIG.SIMULATOR.TURN_ANGLE = 30
_TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = 128
_TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = 128
_TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]

_TASK_CONFIG.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]

_TASK_CONFIG.TASK.SIMPLE_REWARD = CN()
_TASK_CONFIG.TASK.SIMPLE_REWARD.TYPE = "SimpleReward"
_TASK_CONFIG.TASK.SIMPLE_REWARD.SUCCESS_REWARD = 2.5
_TASK_CONFIG.TASK.SIMPLE_REWARD.USE_DTG_REWARD = True
_TASK_CONFIG.TASK.SIMPLE_REWARD.SLACK_PENALTY = -0.01

_TASK_CONFIG.TASK.TYPE = "Nav-v0"
_TASK_CONFIG.TASK.SUCCESS_DISTANCE = 1.0
_TASK_CONFIG.TASK.SUCCESS.SUCCESS_DISTANCE = 1.0

_TASK_CONFIG.TASK.SENSORS = ["IMAGEGOAL_SENSOR"]

_TASK_CONFIG.TASK.MEASUREMENTS = [
    "DISTANCE_TO_GOAL",
    "SUCCESS",
    "SPL",
    "SOFT_SPL",
    "SIMPLE_REWARD",
]


def get_task_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    config = _TASK_CONFIG.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

_CONFIG = _BASE_CONFIG.clone()
_CONFIG.defrost()

_CONFIG.VERBOSE = True

_CONFIG.BASE_TASK_CONFIG_PATH = "configs/tasks/imagenav.yaml"

_CONFIG.TRAINER_NAME = "ppo"
_CONFIG.ENV_NAME = "SimpleRLEnv"
_CONFIG.SENSORS = ["RGB_SENSOR"]

_CONFIG.VIDEO_OPTION = []
_CONFIG.VIDEO_DIR = "data/video"
_CONFIG.TENSORBOARD_DIR = "data/tensorboard"
_CONFIG.EVAL_CKPT_PATH_DIR = "data/checkpoints"
_CONFIG.CHECKPOINT_FOLDER = "data/checkpoints"
_CONFIG.LOG_FILE = "data/train.log"

_CONFIG.NUM_ENVIRONMENTS = 10
_CONFIG.LOG_INTERVAL = 10
_CONFIG.NUM_CHECKPOINTS = 100
_CONFIG.NUM_UPDATES = -1
_CONFIG.TOTAL_NUM_STEPS = 500e6

_CONFIG.FORCE_TORCH_SINGLE_THREADED = True

_CONFIG.RUN_TYPE = None

_CONFIG.EVAL.SPLIT = "val"
_CONFIG.EVAL.USE_CKPT_CONFIG = True

_CONFIG.RL.REWARD_MEASURE = "simple_reward"
_CONFIG.RL.SUCCESS_MEASURE = "success"

_CONFIG.RL.POLICY.name = "EAIPolicy"
_CONFIG.RL.POLICY.backbone = "resnet50"
_CONFIG.RL.POLICY.baseplanes = 32
_CONFIG.RL.POLICY.mask_ratio = 0.5
_CONFIG.RL.POLICY.hidden_size = 512
_CONFIG.RL.POLICY.rnn_type = "GRU"
_CONFIG.RL.POLICY.num_recurrent_layers = 2
_CONFIG.RL.POLICY.use_augmentations = True
_CONFIG.RL.POLICY.pretrained_encoder = None
_CONFIG.RL.POLICY.freeze_backbone = False

_CONFIG.RL.PPO.clip_param = 0.2
_CONFIG.RL.PPO.ppo_epoch = 2
_CONFIG.RL.PPO.num_mini_batch = 2
_CONFIG.RL.PPO.value_loss_coef = 0.5
_CONFIG.RL.PPO.entropy_coef = 0.01
_CONFIG.RL.PPO.lr = 1.25e-4
_CONFIG.RL.PPO.eps = 1e-5
_CONFIG.RL.PPO.max_grad_norm = 0.2
_CONFIG.RL.PPO.num_steps = 64
_CONFIG.RL.PPO.use_gae = True
_CONFIG.RL.PPO.use_linear_lr_decay = False
_CONFIG.RL.PPO.use_linear_clip_decay = False
_CONFIG.RL.PPO.gamma = 0.99
_CONFIG.RL.PPO.tau = 0.95
_CONFIG.RL.PPO.reward_window_size = 50
_CONFIG.RL.PPO.use_normalized_advantage = False
_CONFIG.RL.PPO.hidden_size = 512
_CONFIG.RL.PPO.use_double_buffered_sampler = False

_CONFIG.RL.DDPPO.sync_frac = 0.6
_CONFIG.RL.DDPPO.distrib_backend = "NCCL"


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    config = _CONFIG.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        for k, v in zip(opts[0::2], opts[1::2]):
            if k == "BASE_TASK_CONFIG_PATH":
                config.BASE_TASK_CONFIG_PATH = v

    config.TASK_CONFIG = get_task_config(config.BASE_TASK_CONFIG_PATH)

    if opts:
        config.CMD_TRAILING_OPTS = config.CMD_TRAILING_OPTS + opts
        config.merge_from_list(config.CMD_TRAILING_OPTS)

    if config.NUM_PROCESSES != -1:
        warnings.warn(
            "NUM_PROCESSES is deprecated and will be removed in a future version."
            "  Use NUM_ENVIRONMENTS instead."
            "  Overwriting NUM_ENVIRONMENTS with NUM_PROCESSES for backwards compatibility."
        )

        config.NUM_ENVIRONMENTS = config.NUM_PROCESSES

    config.freeze()
    return config
