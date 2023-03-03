#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

DIST_FRAME_TO_SURFACE = 0.01


def get_ft_radius(finger_type):
    # These are offsets to cube half size for computing contact points
    # Tuned empirically; do not match actual fingertip dimensions
    if finger_type in ["trifingeredu", "trifingernyu", "trifinger_meta"]:
        ft_radius = 0.007  # EDU
    elif finger_type == "trifingerpro":
        ft_radius = 0.008  # PRO
    else:
        raise NameError("Invalid finger_type")

    return ft_radius


def get_finger_base_positions(finger_type):
    """
    The initial position of the fingertips, as angle on the arena, tuned empirically
    These values are critical for good contact point assignment
    """

    if finger_type in ["trifingeredu", "trifingernyu", "trifinger_meta"]:
        theta_0 = 90
        theta_1 = 350
        theta_2 = 220
    elif finger_type == "trifingerpro":
        theta_0 = 90
        theta_1 = 310
        theta_2 = 200
    else:
        raise NameError("Invalid finger_type")

    r = 0.15

    finger_base_positions = [
        np.array(
            [
                [
                    np.cos(theta_0 * (np.pi / 180)) * r,
                    np.sin(theta_0 * (np.pi / 180)) * r,
                    0,
                ]
            ]
        ),
        np.array(
            [
                [
                    np.cos(theta_1 * (np.pi / 180)) * r,
                    np.sin(theta_1 * (np.pi / 180)) * r,
                    0,
                ]
            ]
        ),
        np.array(
            [
                [
                    np.cos(theta_2 * (np.pi / 180)) * r,
                    np.sin(theta_2 * (np.pi / 180)) * r,
                    0,
                ]
            ]
        ),
    ]

    return finger_base_positions
