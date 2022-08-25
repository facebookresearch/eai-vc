import os.path as osp
import shutil

import pytest

from rl_utils.common import Evaluator
from rl_utils.envs import create_vectorized_envs
from rl_utils.interfaces import RandomPolicy


@pytest.mark.parametrize(
    "num_render,traj_save_dir",
    [(0, None), (1, None), (None, None), (0, "data/trajs/traj.pt")],
)
def test_reg_eval(num_render, traj_save_dir):
    if osp.exists("data/vids"):
        shutil.rmtree("data/vids")
    if osp.exists("data/trajs"):
        shutil.rmtree("data/trajs")

    envs = create_vectorized_envs(
        "PointMass-v0",
        32,
    )
    evaluator = Evaluator(
        envs, 0, num_render, "data/vids/", fps=10, save_traj_name=traj_save_dir
    )
    policy = RandomPolicy(envs.action_space)
    ret = evaluator.evaluate(policy, 10, 0)
    assert isinstance(ret, dict)
    assert "episode.r" in ret
