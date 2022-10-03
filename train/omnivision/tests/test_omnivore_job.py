#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import subprocess
import unittest
from pathlib import Path

from omnivision.utils.env import get_buck_mode, is_fbcode


class TestOmnivoreJob(unittest.TestCase):
    def test_omnivore_job(self):
        # WARNING: This test is not run on sandcastle and only runs on OSS
        # or on a devserver
        omnivision_path = Path(__file__).resolve().parents[1]
        omnivore_path = omnivision_path / "projects" / "omnivore"
        if is_fbcode():
            launch_cmd = [
                "buck",
                "run",
                f"@mode/{get_buck_mode()}",
                "//deeplearning/projects/omnivision/projects/omnivore/dev:launch_job",
                "--",
            ]
        else:
            launch_cmd = [
                f"{omnivore_path}/dev/launch_job.py",
            ]
        launch_cmd += [
            "-c",
            "config/experiments/tests/swin_train_synthetic.yaml",
            "--opts",
            "launcher.gpus_per_node",
            "1",
            "--debug",
            "--yes",
        ]
        subprocess.run(
            launch_cmd,
            cwd=omnivore_path,
            check=True,
        )
