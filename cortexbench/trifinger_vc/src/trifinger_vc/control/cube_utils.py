#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

from trifinger_simulation.tasks import move_cube as move_cube_task
import trifinger_vc.control.finger_utils as f_utils

CUBE_HALF_SIZE = move_cube_task._CUBE_WIDTH / 2

# Information about object faces given face_id
OBJ_FACES_INFO = {
    1: {
        "center_param": np.array([0.0, -1.0, 0.0]),
        "face_down_default_quat": np.array([0.707, 0, 0, 0.707]),
        "adjacent_faces": [6, 4, 3, 5],
        "opposite_face": 2,
        "up_axis": np.array([0.0, 1.0, 0.0]),  # UP axis when this face is ground face
    },
    2: {
        "center_param": np.array([0.0, 1.0, 0.0]),
        "face_down_default_quat": np.array([-0.707, 0, 0, 0.707]),
        "adjacent_faces": [6, 4, 3, 5],
        "opposite_face": 1,
        "up_axis": np.array([0.0, -1.0, 0.0]),
    },
    3: {
        "center_param": np.array([1.0, 0.0, 0.0]),
        "face_down_default_quat": np.array([0, 0.707, 0, 0.707]),
        "adjacent_faces": [1, 2, 4, 6],
        "opposite_face": 5,
        "up_axis": np.array([-1.0, 0.0, 0.0]),
    },
    4: {
        "center_param": np.array([0.0, 0.0, 1.0]),
        "face_down_default_quat": np.array([0, 1, 0, 0]),
        "adjacent_faces": [1, 2, 3, 5],
        "opposite_face": 6,
        "up_axis": np.array([0.0, 0.0, -1.0]),
    },
    5: {
        "center_param": np.array([-1.0, 0.0, 0.0]),
        "face_down_default_quat": np.array([0, -0.707, 0, 0.707]),
        "adjacent_faces": [1, 2, 4, 6],
        "opposite_face": 3,
        "up_axis": np.array([1.0, 0.0, 0.0]),
    },
    6: {
        "center_param": np.array([0.0, 0.0, -1.0]),
        "face_down_default_quat": np.array([0, 0, 0, 1]),
        "adjacent_faces": [1, 2, 3, 5],
        "opposite_face": 4,
        "up_axis": np.array([0.0, 0.0, 1.0]),
    },
}


def get_cp_pos_wf_from_cp_param(
    cp_param, obj_pose, cube_half_size=CUBE_HALF_SIZE, ft_radius=0
):
    """
    Compute contact point position in world frame
    Inputs:
    cp_param: Contact point param [px, py, pz]
    cube: Block object, which contains object shape info
    """

    cube_pos_wf = obj_pose["position"]
    cube_quat_wf = obj_pose["orientation"]

    cp = get_cp_of_from_cp_param(cp_param, cube_half_size, ft_radius=ft_radius)

    rotation = Rotation.from_quat(cube_quat_wf)
    translation = np.asarray(cube_pos_wf)

    return rotation.apply(cp["pos_of"]) + translation


def get_cp_pos_wf_from_cp_params(
    cp_params, obj_pose, cube_half_size=CUBE_HALF_SIZE, ft_radius=0
):
    """
    Get contact point positions in world frame from cp_params
    """

    # Get contact points in wf
    fingertip_goal_list = []
    for i in range(len(cp_params)):
        # for i in range(cp_params.shape[0]):
        fingertip_goal_list.append(
            get_cp_pos_wf_from_cp_param(
                cp_params[i], obj_pose, cube_half_size, ft_radius=ft_radius
            )
        )
    return fingertip_goal_list


def get_cp_of_from_cp_param(cp_param, cube_half_size=CUBE_HALF_SIZE, ft_radius=0):
    """
    Compute contact point position in object frame
    Inputs:
    cp_param: Contact point param [px, py, pz]
    """

    effective_cube_half_size = cube_half_size + ft_radius
    obj_shape = (
        effective_cube_half_size,
        effective_cube_half_size,
        effective_cube_half_size,
    )
    cp_of = []
    # Get cp position in OF
    for i in range(3):
        cp_of.append(-obj_shape[i] + (cp_param[i] + 1) * obj_shape[i])

    cp_of = np.asarray(cp_of)

    x_param = cp_param[0]
    y_param = cp_param[1]
    z_param = cp_param[2]
    # For now, just hard code quat
    if y_param == -1:
        quat = (np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2)
    elif y_param == 1:
        quat = (np.sqrt(2) / 2, 0, 0, -np.sqrt(2) / 2)
    elif x_param == 1:
        quat = (0, 0, 1, 0)
    elif z_param == 1:
        quat = (np.sqrt(2) / 2, 0, np.sqrt(2) / 2, 0)
    elif x_param == -1:
        quat = (1, 0, 0, 0)
    elif z_param == -1:
        quat = (np.sqrt(2) / 2, 0, -np.sqrt(2) / 2, 0)

    cp = {"pos_of": cp_of, "quat_of": quat}
    return cp


def get_face_from_cp_param(cp_param):
    """
    Get face id on cube, given cp_param
    cp_param: [x,y,z]
    """
    x_param = cp_param[0]
    y_param = cp_param[1]
    z_param = cp_param[2]
    # For now, just hard code quat
    if y_param == -1:
        face = 1
    elif y_param == 1:
        face = 2
    elif x_param == 1:
        face = 3
    elif z_param == 1:
        face = 4
    elif x_param == -1:
        face = 5
    elif z_param == -1:
        face = 6

    return face


def get_cp_params(obj_pose, finger_type):
    """
    Get contact points on cube for each finger
    Assign closest cube face to each finger
    Since we are lifting object, don't worry about wf z-axis, just care about wf xy-plane
    """

    # face that is touching the ground
    ground_face = get_closest_ground_face(obj_pose)

    # get finger base positions
    finger_base_positions = f_utils.get_finger_base_positions(finger_type)

    # Transform finger base positions to object frame
    base_pos_list_of = []
    for f_wf in finger_base_positions:
        f_of = get_of_from_wf(f_wf, obj_pose)
        base_pos_list_of.append(f_of)

    # Find distance from x axis and y axis, and store in xy_distances
    # Need some additional logic to prevent multiple fingers from being assigned to same face
    x_axis = np.array([1, 0])
    y_axis = np.array([0, 1])

    # Object frame axis corresponding to plane parallel to ground plane
    x_ind, y_ind = _get_parallel_ground_plane_xy(ground_face)

    xy_distances = np.zeros(
        (3, 2)
    )  # Row corresponds to a finger, columns are x and y axis distances
    for f_i, f_of in enumerate(base_pos_list_of):
        point_in_plane = np.array(
            [f_of[0, x_ind], f_of[0, y_ind]]
        )  # Ignore dimension of point that's not in the plane
        x_dist = _get_distance_from_pt_2_line(x_axis, np.array([0, 0]), point_in_plane)
        y_dist = _get_distance_from_pt_2_line(y_axis, np.array([0, 0]), point_in_plane)

        xy_distances[f_i, 0] = x_dist
        xy_distances[f_i, 1] = y_dist

    # Do the face assignment - greedy approach (assigned closest fingers first)
    free_faces = OBJ_FACES_INFO[ground_face][
        "adjacent_faces"
    ].copy()  # List of face ids that haven't been assigned yet
    assigned_faces = np.zeros(3)
    for i in range(3):
        # Find indices max element in array
        max_ind = np.unravel_index(np.argmax(xy_distances), xy_distances.shape)
        curr_finger_id = max_ind[0]
        furthest_axis = max_ind[1]

        # print("current finger {}".format(curr_finger_id))
        # Do the assignment
        x_dist = xy_distances[curr_finger_id, 0]
        y_dist = xy_distances[curr_finger_id, 1]
        if furthest_axis == 0:  # distance to x axis is greater than to y axis
            if base_pos_list_of[curr_finger_id][0, y_ind] > 0:
                face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][1]  # 2
            else:
                face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][0]  # 1
        else:
            if base_pos_list_of[curr_finger_id][0, x_ind] > 0:
                face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][2]  # 3
            else:
                face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][3]  # 5
        # print("first choice face: {}".format(face))

        # Handle faces that may already be assigned
        if face not in free_faces:
            alternate_axis = abs(furthest_axis - 1)
            if alternate_axis == 0:
                if base_pos_list_of[curr_finger_id][0, y_ind] > 0:
                    face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][1]  # 2
                else:
                    face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][0]  # 1
            else:
                if base_pos_list_of[curr_finger_id][0, x_ind] > 0:
                    face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][2]  # 3
                else:
                    face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][3]  # 5
            # print("second choice face: {}".format(face))

        # If backup face isn't free, assign random face from free_faces
        if face not in free_faces:
            # print("random")
            # print(xy_distances[curr_finger_id, :])
            face = free_faces[0]
        assigned_faces[curr_finger_id] = face

        # Replace row with -np.inf so we can assign other fingers
        xy_distances[curr_finger_id, :] = -np.inf
        # Remove face from free_faces
        free_faces.remove(face)
    # print(assigned_faces)
    # Set contact point params
    cp_params = []
    for i in range(3):
        face = assigned_faces[i]
        param = OBJ_FACES_INFO[face]["center_param"].copy()
        # print(i)
        # print(param)
        cp_params.append(param)
    # print("assigning cp params for lifting")
    # print(cp_params)
    return cp_params


def get_closest_ground_face(obj_pose):
    """
    Determine face that is closest to ground
    """

    min_z = np.inf
    min_face = None
    for i in range(1, 7):
        c = OBJ_FACES_INFO[i]["center_param"].copy()
        c_wf = get_wf_from_of(c, obj_pose)
        if c_wf[2] < min_z:
            min_z = c_wf[2]
            min_face = i

    return min_face


def get_vertices_wf(obj_pose):
    """Get vertices of cube in world frame, given obj_pose in world frame"""

    v_of_dict = get_vertices_of()
    v_wf_dict = {}

    # TODO fill this in
    for k, v_of in v_of_dict.items():
        v_wf = get_wf_from_of(v_of, obj_pose)
        v_wf_dict[k] = v_wf

    return v_wf_dict


def get_vertices_of():
    """Get vertices of cube in object frame"""

    v = {
        0: np.array([-1, -1, -1]) * CUBE_HALF_SIZE,
        1: np.array([1, -1, -1]) * CUBE_HALF_SIZE,
        2: np.array([-1, -1, 1]) * CUBE_HALF_SIZE,
        3: np.array([1, -1, 1]) * CUBE_HALF_SIZE,
        4: np.array([-1, 1, -1]) * CUBE_HALF_SIZE,
        5: np.array([1, 1, -1]) * CUBE_HALF_SIZE,
        6: np.array([-1, 1, 1]) * CUBE_HALF_SIZE,
        7: np.array([1, 1, 1]) * CUBE_HALF_SIZE,
    }

    return v


##############################################################################
# Transformation functions
##############################################################################


def get_wf_from_of(p, obj_pose):
    """
    Trasform point p from world frame to object frame, given object pose
    """
    cube_pos_wf = obj_pose["position"]
    cube_quat_wf = obj_pose["orientation"]

    rotation = Rotation.from_quat(cube_quat_wf)
    translation = np.asarray(cube_pos_wf)

    return rotation.apply(p) + translation


def get_of_from_wf(p, obj_pose):
    """
    Trasform point p from object frame to world frame, given object pose
    """
    cube_pos_wf = obj_pose["position"]
    cube_quat_wf = obj_pose["orientation"]

    rotation = Rotation.from_quat(cube_quat_wf)
    translation = np.asarray(cube_pos_wf)

    rotation_inv = rotation.inv()
    translation_inv = -rotation_inv.apply(translation)

    return rotation_inv.apply(p) + translation_inv


##############################################################################
# Non-cube specific functions TODO move somewhere else
##############################################################################


def lin_interp_pos_two_points(x_cur, x_des, T, time_step=0.001):
    """
    Linearly interpolate x_cur, x_des positions to get waypoints
    No orientation

    args:
        x_cur: start position
        x_des: end position
        T: duration of trajectory, in seconds
        time_step: timestep between waypoints (simulation timestep)
    """

    delta_x = x_des - x_cur
    dx = delta_x / T
    num_points = int(T / time_step)

    x_traj = np.linspace(x_cur, x_des, num=num_points)
    dx_traj = np.tile(dx, (num_points, 1))

    return x_traj, dx_traj


def lin_interp_pos(x, time_step_in, time_step_out=0.001):
    """
    Linearly interpolate between all position waypoints in x (between each row) [T, dim]
    """

    T = x.shape[0]
    interp_n = int(
        time_step_in / time_step_out
    )  # Number of interpolation points between two waypoints

    # Linearly interpolate between each position waypoint (row) and force waypoint
    # Initial row indices
    row_ind_in = np.arange(T)
    # Output row coordinates
    row_coord_out = np.linspace(0, T - 1, interp_n * (T - 1) + T)

    # scipy.interpolate.interp1d instance
    itp_x = interp1d(row_ind_in, x, axis=0)

    x_interpolated = itp_x(row_coord_out)
    return x_interpolated


def lin_interp_pos_traj(x_in, time_step_in, time_step_out):
    """
    Linearly interpolate between all position waypoints in x (between each row) [T, dim]
    Output position and velocity trajectories
    """

    x_pos_traj = lin_interp_pos(x_in, time_step_in, time_step_out)

    x_vel_traj = np.zeros(x_pos_traj.shape)
    for i in range(x_pos_traj.shape[0] - 1):
        v = (x_pos_traj[i + 1, :] - x_pos_traj[i, :]) / time_step_out
        x_vel_traj[i, :] = v

    return x_pos_traj, x_vel_traj


##############################################################################
# Private functions
##############################################################################


def _get_parallel_ground_plane_xy(ground_face):
    """
    Given a ground face id, get the axes that are parallel to the floor
    """
    if ground_face in [1, 2]:
        x_ind = 0
        y_ind = 2
    if ground_face in [3, 5]:
        x_ind = 2
        y_ind = 1
    if ground_face in [4, 6]:
        x_ind = 0
        y_ind = 1
    return x_ind, y_ind


def _get_distance_from_pt_2_line(a, b, p):
    """
    Get distance from point to line (in 2D)
    Inputs:
    a, b: points on line
    p: standalone point, for which we want to compute its distance to line
    """
    a = np.squeeze(a)
    b = np.squeeze(b)
    p = np.squeeze(p)

    ba = b - a
    ap = a - p
    c = ba * (np.dot(ap, ba) / np.dot(ba, ba))
    d = ap - c

    return np.sqrt(np.dot(d, d))
