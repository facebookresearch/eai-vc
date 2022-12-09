from dotmap import DotMap
import numpy as np
import pinocchio as pin
from cto.objects import Cube, Plate
from cto.trajectory import generate_trajectories


def get_default_params(object_urdf, box_type="cube"):
    params = DotMap()

    # MCTS
    params.err_threshold = 0.03
    params.classifier_threshold = 0.5

    # object
    params.box_type = box_type
    if box_type == "cube":
        length = 0.065
        box = Cube(length, object_urdf)
        params.n_surfaces = 6
        params.forbidden_surfaces = set([5, 6])
    if box_type == "plate":
        box = Plate(0.05, 0.1, 0.02, object_urdf)
        params.n_surfaces = 4
        params.forbidden_surfaces = set([])

    params.object_urdf = object_urdf
    params.box = box
    params.mass = box.mass
    params.inertia = box.inertia
    params.simplices = [box.get_simplices(i) for i in range(params.n_surfaces)]
    params.contact_frame_orientation = [
        box.get_contact_frame(i) for i in range(params.n_surfaces)
    ]

    # environment
    params.gravity = np.array([0, 0, -9.81])
    params.dt = 0.1
    params.environment_friction = 0.8
    params.ground_height = 0
    params.box_com_height = params.ground_height + box.height / 2

    # contact
    params.n_contacts = 3
    params.contact_duration = 8

    # friction
    params.friction_coeff = 0.8

    # bounds
    params.max_location = 0.05
    params.max_force = 10

    return params


def update_params(params, desired_poses, repr="xyzaxisangle"):
    params.n_desired_poses = len(desired_poses)

    # poses
    if repr == "xyzaxisangle":
        params.desired_poses = [
            pin.SE3(pin.exp3(pose[3:]), pose[:3]) for pose in desired_poses
        ]
    elif repr == "xyzquat":
        params.desired_poses = [pin.XYZQUATToSE3(pose) for pose in desired_poses]
    elif repr == "SE3":
        params.desired_poses = desired_poses

    params.pose_start = params.desired_poses[0]
    params.pose_end = params.desired_poses[-1]
    params.pose_diff = pin.log6(params.pose_start.actInv(params.pose_end))

    # interpolation
    params.n_modes_static = 3
    params.n_modes_dynamic = 7
    params.n_modes_segment = params.n_modes_static + params.n_modes_dynamic
    params.n_contact_modes = params.n_modes_segment * (params.n_desired_poses - 1)

    traj_desired = generate_trajectories(params)

    params.traj_desired = traj_desired
    params.horizon = traj_desired.horizon
    params.environment_contacts = traj_desired.environment_contacts

    return DotMap(params)
