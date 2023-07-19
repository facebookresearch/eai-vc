#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List
from setuptools import setup
from setuptools import find_packages
from setuptools import find_namespace_packages

def get_file_content(file_path: str, *args, **kwargs) -> str:
    """
    Read the content of a file.
        Args:
            file_path: Path to the file.
            *args: Additional arguments to pass to open().
            **kwargs: Additional keyword arguments to pass to open().
        Returns:
            The content of the file.
    """
    with open(file_path, *args, **kwargs) as f:
        return f.read()

def get_package_version() -> str:
    """
    Get the version of the package. The version is defined in the file version.py.
    Returns:
        The version of the package.
    """
    import os.path as osp
    import sys

    sys.path.insert(0, osp.join(osp.dirname(__file__), "src", "vc_models"))
    from version import VERSION

    return VERSION

def parse_sections_md_file(file_path: str) -> Dict[str, str]:
    """
    Parse a markdown file into sections.

    Sections are defined by the first level 1 or 2 header.

    Args:
        file_path: Path to the markdown file.

    Returns:
        A dictionary of sections with the section content.
    """
    with open(file_path, "r") as file:
        content = file.read()

    section_dict: Dict[str, str] = {}
    lines = content.split("\n")
    current_title = ""
    code_block_mode = False

    for line in lines:
        if line.startswith("```"):
            code_block_mode = not code_block_mode
        elif code_block_mode:
            section_dict[current_title] += f"{line}\n"
        elif line.startswith("# "):
            current_title = line[2:]
            section_dict[current_title] = f"{line}\n"
        elif line.startswith("## "):
            current_title = line[3:]
            section_dict[current_title] = f"{line}\n"
        elif current_title:
            section_dict[current_title] += f"{line}\n"

    return section_dict

desc_sections = parse_sections_md_file("../README.md")

sections_to_include = [
    "Visual Cortex and CortexBench",
    "Open-Sourced Models",
    "Load VC-1",
    "Citing Visual Cortex",
    "License",
]
long_description = "".join(desc_sections[section] for section in sections_to_include)
long_description = long_description.replace("# Visual Cortex and CortexBench", "# Visual Cortex").replace("vc1_teaser.gif", "vc1_teaser_small.gif").replace("./", "https://github.com/facebookresearch/eai-vc/tree/main/")

packages_to_release = find_packages(where="src") + find_namespace_packages(
    include=["hydra_plugins.*"], where="src")

if __name__ == "__main__":
    setup(
        name="vc_models",
        install_requires=get_file_content("requirements.txt").strip().split("\n"),
        packages=packages_to_release,
        version=get_package_version(),
        package_dir={"": "src"},
        include_package_data=True,
        package_data={'vc_models': ['conf/model/*.yaml']},
        description="Visual Cortex Models: A lightweight package for loading cutting-edge efficient Artificial Visual Cortex models for Embodied AI applications.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Meta AI Research",
        license="CC-BY-NC License",
        url="https://eai-vc.github.io/",
        project_urls={
            "GitHub repo": "https://github.com/facebookresearch/eai-vc",
            "Bug Tracker": "https://github.com/facebookresearch/eai-vc/issues",
        },
        classifiers=[
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "Development Status :: 5 - Production/Stable",
            "License :: Other/Proprietary License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.8",
            "Operating System :: OS Independent",
            "Natural Language :: English",
        ],
    )