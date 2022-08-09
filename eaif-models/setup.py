from setuptools import setup
from setuptools import find_packages
from setuptools import find_namespace_packages


packages = find_packages(where="src") + find_namespace_packages(
    include=["hydra_plugins.*"], where="src"
)
install_requires = [
    "torch >= 1.10.2",
    "torchvision >= 0.11.3",
    "timm==0.6.5",
    "hydra-core",
    "wandb",
]

setup(
    name="eaif-models",
    version="0.1",
    packages=packages,
    package_dir={"": "src"},
    install_requires=install_requires,
    include_package_data=True,
)
