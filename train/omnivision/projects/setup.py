#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from setuptools import find_packages, setup

setup(
    name="omnivision_projects",
    version="0.0.1",
    license="Apache 2.0",
    author="Facebook AI",
    url="https://github.com/fairinternal/omnivision",
    description="Omnivision projects",
    python_requires=">=3.8",
    install_requires=[
        # omnivision is a core dependency
        "omnivision",
        # project specific dependencies
        "einops",
        "ftfy",
        "regex",
        "timm",
        "webdataset",
    ],
    packages=find_packages(),
)
