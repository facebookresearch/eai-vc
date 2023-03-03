#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from setuptools import find_packages


install_requires = [
    "hydra-core",
    "wandb",
    "gym",
]

setup(
    name="trifinger",
    version="0.1",
    install_requires=install_requires,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
