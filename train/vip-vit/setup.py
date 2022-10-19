# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
from setuptools import setup, find_packages

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))
    
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='vip_vit',
    version='0.0.0',
    packages=find_packages(),
    description='VIP: Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training',
    long_description=read('README.md'),
    author='Jason Ma (Meta AI)',
    # install_requires=[
    # 'gdown==4.4.0', 
    # 'torch<=1.10.2,>=1.7.1',
    # 'torchvision<=0.11.3,>=0.8.2',
    # 'omegaconf==2.1.1',
    # 'hydra-core==1.1.1',
    # 'pillow==9.0.1',
    # 'opencv-python',
    # 'matplotlib',
    # 'flatten_dict',
    # 'tabulate',
    # 'pandas',
    # 'scipy',
    # 'scikit-learn',
    # 'scikit-video',
    # 'transforms3d',
    # 'moviepy',
    # 'termcolor', 
    # ]
)
