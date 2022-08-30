from setuptools import setup
from setuptools import find_packages


install_requires = [
    "hydra-core",
    "wandb",
    "mujoco-py",
    "mjrl",
    "gym",
    "mj_envs",
    "dmc2gym",
]

setup(
    name="eaif_mujoco",
    version="0.1",
    install_requires=install_requires,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
