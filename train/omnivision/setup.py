#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from setuptools import find_packages, setup

setup(
    name="omnivision",
    version="0.0.1",
    license="Apache 2.0",
    author="Facebook AI",
    url="https://github.com/fairinternal/omnivision",
    description="Omnivision trainer",
    python_requires=">=3.8",
    install_requires=[
        "hydra-core",
        "submitit>=1.4.4",
        "pytorchvideo>=0.1.5",
        "fvcore",
        "opencv-python",
        "tensorboard==2.9.1",
        "torch>=1.12",
        "torchvision>=0.13",
        "fasttext>=0.9.2",
        "fairscale==0.4.6",
    ],
    extras_require={
        "dev": [
            "sphinx",
            ##################################
            # Formatter settings based on
            # `pyfmt -V`
            "black==22.3.0",
            "ufmt==2.0.0b2",
            "usort==1.0.2",
            "libcst==0.4.1",
            ##################################
        ],
        "omnivore": [
            "torchaudio>=0.10.0+cu111",
            "webdataset==0.2.26",
            "einops",
            "ftfy",
            "regex",
            "timm",
        ],
        "omniscale": [],
    },
    packages=find_packages(exclude=("scripts", "tests")),
)
