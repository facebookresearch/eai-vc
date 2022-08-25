import os.path as osp

import setuptools

cur_dir = osp.dirname(osp.realpath(__file__))
requirementPath = osp.join(cur_dir, "requirements.txt")
install_requires = []
if osp.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setuptools.setup(
    name="rl-utils",
    version="0.1",
    author="Andrew Szot",
    author_email="me@andrewszot.com",
    description="Library for RL research.",
    url="https://github.com/aszot/rl-utils",
    install_requires=install_requires,
    packages=setuptools.find_packages(),
)
