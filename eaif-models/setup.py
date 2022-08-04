from setuptools import setup
from setuptools import find_packages

packages = find_packages(where="src")
install_requires = [
    "torch >= 1.10.2",
    "torchvision >= 0.11.3",
    "timm==0.6.5",
    "hydra-core",
]

setup(
    name="eaif-models",
    version="0.1",
    packages=packages,
    package_dir={"": "src"},
    install_requires=install_requires,
)
