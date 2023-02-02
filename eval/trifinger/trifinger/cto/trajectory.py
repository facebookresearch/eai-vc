import pinocchio as pin
import numpy as np
from dotmap import DotMap


def location_to_world(location, pose_ref):
    p = pose_ref.translation
    R = pose_ref.rotation
    return p + R @ location


def location_traj_to_world(traj, traj_ref):
    traj_world = np.zeros_like(traj)
    for n in range(len(traj)):
        pose_ref = pin.XYZQUATToSE3(traj_ref[n])
        traj_world[n] = location_to_world(traj[n], pose_ref)
    return traj_world


def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return r, az, el


def sph2cart(r, az, el):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z


def interpolate_rigid_motion(diff, duration, t):
    # first we compute the coefficients
    a5 = 6 / (duration**5)
    a4 = -15 / (duration**4)
    a3 = 10 / (duration**3)

    # now we compute s and ds/dt
    s = a3 * t**3 + a4 * t**4 + a5 * t**5
    ds = 3 * a3 * t**2 + 4 * a4 * t**3 + 5 * a5 * t**4
    dds = 6 * a3 * t + 12 * a4 * t**2 + 20 * a5 * t**3

    # now we compute q and dq/dt
    q = pin.exp6(s * diff)
    dq = diff * ds
    ddq = diff * dds

    return q, dq, ddq


def interpolate_trajectory_sphere(loc_start, loc_end, horizon, dt, pose_only=True):
    loc_start_sph = np.array(cart2sph(*loc_start))
    loc_end_sph = np.array(cart2sph(*loc_end))

    # wrap the angle difference to be less than pi
    loc_diff = loc_end_sph - loc_start_sph
    loc_diff_wraped = (loc_diff + np.pi) % (2 * np.pi) - np.pi
    loc_end_sph = loc_start_sph + loc_diff_wraped

    traj_sph = interpolate_trajectory_line(loc_start_sph, loc_end_sph, horizon, dt)
    # convert sphr coord back to cart
    q = np.zeros((horizon + 1, 3))
    for n in range(horizon + 1):
        q[n] = np.array(sph2cart(*traj_sph[n]))

    # better convert velocity in the sphr coord to cart but let's just use finite diff
    dq = np.gradient(q, axis=0) / dt
    ddq = np.gradient(dq, axis=0) / dt

    if pose_only:
        return q

    traj_cart = DotMap()
    traj_cart.q = q
    traj_cart.dq = dq
    traj_cart.ddq = ddq
    return traj_cart


def interpolate_trajectory_line(loc_start, loc_end, horizon, dt, pose_only=True):
    pose_start = pin.SE3(np.eye(3), loc_start)
    loc_diff = loc_end - loc_start
    pose_diff = pin.Motion(loc_diff, np.zeros(3))

    q = np.zeros((horizon + 1, 3))
    dq = np.zeros((horizon + 1, 3))
    ddq = np.zeros((horizon + 1, 3))

    for n in range(horizon + 1):
        qn, dqn, ddqn = interpolate_rigid_motion(pose_diff, horizon * dt, n * dt)
        q[n] = pin.SE3ToXYZQUAT(pose_start.act(qn))[:3]
        dq[n] = dqn.vector[:3]
        ddq[n] = ddqn.vector[:3]

    if pose_only:
        return q

    traj = DotMap()
    traj.q, traj.dq, traj.ddq = q, dq, ddq

    return traj


def combine_traj(traj1, traj2, pose_only=True):
    if pose_only:
        return np.vstack((traj1, traj2))
    traj_combined = DotMap()
    traj_combined.q = np.vstack((traj1.q, traj2.q))
    traj_combined.dq = np.vstack((traj1.dq, traj2.dq))
    traj_combined.ddq = np.vstack((traj1.ddq, traj2.ddq))
    return traj_combined


def interpolate_trajectory(pose_start, pose_end, horizon, dt, pose_only=True):
    pose_diff = pin.log6(pose_start.actInv(pose_end))
    q = np.zeros((horizon + 1, 7))
    dq = np.zeros((horizon + 1, 6))
    ddq = np.zeros((horizon + 1, 6))

    for n in range(horizon + 1):
        qn, dqn, ddqn = interpolate_rigid_motion(pose_diff, horizon * dt, n * dt)
        q[n] = pin.SE3ToXYZQUAT(pose_start.act(qn))
        dq[n] = dqn.vector
        ddq[n] = ddqn.vector

    if pose_only:
        return q
    traj = DotMap()
    traj.q, traj.dq, traj.ddq = q, dq, ddq

    return traj


def generate_trajectories(params, dt_ratio=1, pose_only=False):
    d = params.contact_duration
    d_static = params.n_modes_static * d * dt_ratio
    d_dynamic = params.n_modes_dynamic * d * dt_ratio
    d_segment = d_static + d_dynamic

    horizon = (len(params.desired_poses) - 1) * d_segment

    q = np.zeros((horizon, 7))
    dq = np.zeros((horizon, 6))
    ddq = np.zeros((horizon, 6))

    for i in range(len(params.desired_poses) - 1):
        pose_start, pose_end = params.desired_poses[i], params.desired_poses[i + 1]

        q[i * d_segment : i * d_segment + d_static, :] = pin.SE3ToXYZQUAT(pose_start)

        traj = interpolate_trajectory(
            pose_start, pose_end, d_dynamic - 1, params.dt / dt_ratio, pose_only=False
        )

        q[i * d_segment + d_static : (i + 1) * d_segment, :] = traj.q
        dq[i * d_segment + d_static : (i + 1) * d_segment, :] = traj.dq
        ddq[i * d_segment + d_static : (i + 1) * d_segment, :] = traj.ddq

    if pose_only:
        return q

    traj = DotMap()
    traj.q = q
    traj.dq = dq
    traj.ddq = ddq
    traj.horizon = horizon
    traj.total_force, traj.total_torque = get_wrench(traj, params)
    traj.environment_contacts = get_environment_contacts(traj, params)

    return traj


def get_wrench(traj, params):
    """
    get wrench in the body frame, excluding gravity
    """
    force = np.zeros((traj.horizon, 3))
    torque = np.zeros((traj.horizon, 3))

    for n in range(traj.horizon):
        curr_pose = pin.XYZQUATToSE3(traj.q[n])
        gravity_body = curr_pose.rotation.T @ (params.mass * params.gravity)
        force[n] = (
            params.mass * (traj.ddq[n, :3] + np.cross(traj.dq[n, 3:], traj.dq[n, :3]))
            - gravity_body
        )
        torque[n] = params.inertia @ traj.ddq[n, 3:] + np.cross(
            traj.dq[n, 3:], params.inertia @ traj.dq[n, 3:]
        )

    return force, torque


def get_environment_contacts(traj, params):
    """
    Define environment contacts
    s: world frame
    b: object frame
    c: contact frame
    contact frame orientation relative to the world frame
    TODO: do proper collision detection
    """

    Rsc = np.eye(3)
    from cto.mcts.contacts import environment_contact

    environment_contacts = [None] * traj.horizon

    if params.box_type == "cube":
        environment_contacts_locations = [
            params.box.width / 2 * np.array([1.0, -1.0, -1.0]),
            params.box.width
            / 2
            * np.array(
                [
                    1.0,
                    1.0,
                    -1,
                ]
            ),
            params.box.width / 2 * np.array([-1.0, -1.0, -1.0]),
            params.box.width / 2 * np.array([-1.0, 1.0, -1.0]),
        ]
    elif params.box_type == "plate":
        environment_contacts_locations = [
            np.array([0.025, -0.05, -0.01]),
            np.array([0.025, 0.05, -0.01]),
            np.array([-0.025, -0.05, -0.01]),
            np.array([-0.025, 0.05, -0.01]),
        ]

    for n in range(traj.horizon):
        curr_pose = pin.XYZQUATToSE3(traj.q[n])
        Rsb = curr_pose.rotation
        psb = curr_pose.translation
        Rbc = Rsb.T @ Rsc
        curr_environment_contacts = []
        w_body = traj.dq[n][3:]
        for loc in environment_contacts_locations:
            loc_world = Rsb @ loc + psb
            if (
                abs(loc_world[-1] - params.ground_height) <= 1e-3
            ):  # check if the contact location is touching the table surface
                v_body = np.cross(w_body, loc) + traj.dq[n][:3]
                v_contact = (Rbc.T @ v_body)[:2]
                if np.allclose(v_contact, 0):
                    curr_environment_contacts.append(
                        environment_contact(loc, Rbc, params, type="sticking")
                    )
                else:
                    curr_environment_contacts.append(
                        environment_contact(
                            loc,
                            Rbc,
                            params,
                            type="sliding",
                            sliding_direction=v_contact,
                        )
                    )

        environment_contacts[n] = curr_environment_contacts
    return environment_contacts


def get_end_effector_regions(traj, params):
    """
    it's easier to specify the end-effector workspace in the world frame
    but we need the constraints on r in the body frame

        plug
        r_world = R @ r_body + p
        into
        A_world @ r_world <= b_world

        =>

        A_world @ R @ r_body + A_world @ p <= b_world
        compare
        A_body @ r_body <= b_body

        =>

        A_body = A_world @ R
        b_body = b_world - A_world @ p
    """

    # finger 1
    A1_world = np.zeros((3, 3))
    A1_world[0, 0] = 1
    b1_world = np.zeros(3)
    b1_world[0] = 0.1

    # finger 2
    A2_world = np.zeros((3, 3))
    A2_world[0, 0] = -1
    b2_world = np.zeros(3)
    b2_world[0] = 0.1

    end_effector_regions = []
    for n in range(params.horizon):
        end_effector_region = DotMap()
        curr_pose = pin.XYZQUATToSE3(traj.q[n])
        R = curr_pose.rotation
        p = curr_pose.translation
        Ar1 = A1_world @ R
        Ar2 = A2_world @ R
        br1 = b1_world - A1_world @ p
        br2 = b2_world - A2_world @ p
        end_effector_region.Ar = [Ar1, Ar2]
        end_effector_region.br = [br1, br2]
        end_effector_regions.append(end_effector_region)

    return end_effector_regions


def integrate_trajectory(pose_start, idx_start, idx_end, force, torque, params):
    traj_length = idx_end - idx_start
    q = np.zeros((traj_length, 7))
    dq = np.zeros((traj_length, 6))
    ddq = np.zeros((traj_length, 6))
    q[0] = pose_start
    poses = [pin.XYZQUATToSE3(pose_start)]
    q_des = params.traj_desired.q[idx_start:idx_end]

    for n in range(traj_length - 1):
        curr_pose = poses[n]
        desired_pose = pin.XYZQUATToSE3(q_des[n])
        gravity_body = desired_pose.rotation.T @ (params.mass * params.gravity)
        total_force = force[n] + gravity_body
        ddq[n, :3] = 1 / params.mass * total_force - np.cross(dq[n, 3:], dq[n, :3])
        ddq[n, 3:] = np.linalg.pinv(params.inertia) @ (
            torque[n] - np.cross(dq[n, 3:], params.inertia @ dq[n, 3:])
        )
        ddq[n][np.abs(ddq[n]) < 1e-3] = 0
        dq[n + 1] = dq[n] + params.dt * ddq[n]
        next_pose = curr_pose.act(pin.exp6(dq[n + 1] * params.dt))
        poses.append(next_pose)
        q[n + 1] = pin.SE3ToXYZQUAT(next_pose)

    traj = DotMap()
    traj.poses = poses
    traj.q, traj.dq, traj.ddq = q, dq, ddq
    traj.total_force, traj.total_torque = force, torque

    return traj


def generate_random_poses(
    n_desired_poses, lb, ub, diff_lb, diff_ub, init_lb=None, init_ub=None
):
    if init_lb is None:
        init_lb = lb
    if init_ub is None:
        init_ub = ub

    desired_poses = [np.random.uniform(init_lb, init_ub)]

    for i in range(n_desired_poses):
        while True:
            diff = np.random.uniform(diff_lb, diff_ub)
            pose = desired_poses[-1] + diff
            if all(pose >= lb) and all(pose <= ub):
                break
        desired_poses.append(pose)

    return desired_poses


# hard-coded motion generator for the NYU double figner and a 10x10x10 cm Cube
# TODO: replace this with a proper RRT planner
def generate_ee_motion(state, sol, dt_sim, dt_plan, params):
    def get_simplices(surface, r=0.12):
        # hard-coded surface vertices for a cube
        if params.box_type == "cube":
            corners = np.array([[-r, -r], [r, -r], [-r, r], [r, r]])
            simplices = []
            for i in range(3):
                simplices.append(np.insert(corners, i, -r, axis=1))
                simplices.append(np.insert(corners, i, r, axis=1))
            return simplices[surface]
        elif params.box_type == "plate":
            # hard-coded surface vertices for a plate
            w = 0.1
            l = 0.18
            h = 0.05
            corners1 = np.array([[-w, 0.6 * l], [w, 0.6 * l], [-w, l], [w, l]])
            corners2 = np.array([[-w, -l], [w, -l], [-w, -0.6 * l], [w, -0.6 * l]])
            simplices = []
            simplices.append(np.insert(corners1, 2, -h, axis=1))
            simplices.append(np.insert(corners1, 2, h, axis=1))
            simplices.append(np.insert(corners2, 2, -h, axis=1))
            simplices.append(np.insert(corners2, 2, h, axis=1))
            return simplices[surface]

    dt_ratio = int(dt_plan / dt_sim)

    box_traj = generate_trajectories(params, dt_ratio=dt_ratio, pose_only=True)

    forces0 = [forces[0] for forces in sol.forces_world]
    forces0_r = np.repeat(forces0, repeats=100, axis=0)

    forces1 = [forces[1] for forces in sol.forces_world]
    forces1_r = np.repeat(forces1, repeats=100, axis=0)

    forces_r = [forces0_r, forces1_r]

    rest_locations = [[0, 0] for i in range(len(state))]
    if params.box_type == "cube":
        rest_locations[0] = [np.array([-0.05, 0, 0.12]), np.array([0.05, 0, 0.12])]
    elif params.box_type == "plate":
        rest_locations[0] = [np.array([-0.05, 0, 0.05]), np.array([0.05, 0, 0.05])]

    switch = [[0, 0] for i in range(len(state))]
    trajs = [[None, None] for i in range(len(state))]
    forces = [[None, None] for i in range(len(state))]
    way_points = [[[], []] for i in range(len(state))]
    d = params.contact_duration

    for i in range(len(state)):
        prev_id = i - 1 if i > 0 else i
        next_id = i + 1 if i < len(state) - 1 else i
        h = dt_ratio * d

        for c in range(params.n_contacts):
            if state[i][c] == state[next_id][c]:
                if state[i][c] == 0:
                    # not in contact no switch, remains at the rest location
                    switch[i][c] = 0
                    loc = rest_locations[i][c]
                    forces[i][c] = np.zeros((h, 3))
                else:
                    # in contact no switch, follow the planned location traj
                    switch[i][c] = 1
                    loc = sol.locations[i * d][c]
                    forces[i][c] = forces_r[c][h * i : h * (i + 1)]

                way_points[i][c] += [loc, loc]
                traj_body = interpolate_trajectory_line(
                    loc, loc, d * dt_ratio - 1, dt_sim
                )

                # convert to world frame & save traj/force
                traj_ref = box_traj[h * i : h * (i + 1)]
                trajs[i][c] = location_traj_to_world(traj_body, traj_ref)

            if state[i][c] != state[next_id][c]:
                if state[next_id][c] != 0:
                    # establishing contact
                    # previous rest location -> outer cube -> inner cube -> planned location traj
                    switch[i][c] = 2
                    next_surface = state[next_id][c]
                    loc1 = rest_locations[i][c]
                    loc2 = np.mean(get_simplices(next_surface - 1), axis=0)
                    loc3 = sol.locations[next_id * d][c]
                    traj1_ref = traj2_ref = box_traj[h * i : h * (i + 1)]
                    traj1_body = interpolate_trajectory_sphere(
                        loc1, loc2, d * dt_ratio - 1, dt_sim
                    )
                    forces[i][c] = np.zeros((2 * h, 3))
                else:
                    # breaking contact
                    # complete the planned traj -> outer cube (rest location)
                    switch[i][c] = 3
                    curr_surface = state[i][c]
                    loc1 = loc2 = sol.locations[i * d][c]
                    loc3 = np.mean(get_simplices(curr_surface - 1), axis=0)

                    traj1_body = interpolate_trajectory_line(
                        loc1, loc2, d * dt_ratio - 1, dt_sim
                    )
                    traj1_ref = box_traj[h * i : h * (i + 1)]
                    ref_end = pin.XYZQUATToSE3(traj1_ref[-1])
                    traj2_ref = interpolate_trajectory(
                        ref_end, ref_end, d * dt_ratio - 1, dt_sim
                    )
                    forces[i][c] = np.vstack(
                        (forces_r[c][h * i : h * (i + 1)], np.zeros((h, 3)))
                    )

                way_points[i][c] += [loc1, loc2, loc3]
                traj2_body = interpolate_trajectory_line(
                    loc2, loc3, d * dt_ratio - 1, dt_sim
                )

                # convert to world frame
                traj1_world = location_traj_to_world(traj1_body, traj1_ref)
                traj2_world = location_traj_to_world(traj2_body, traj2_ref)

                # save traj/force
                trajs[i][c] = combine_traj(traj1_world, traj2_world)

            rest_locations[next_id][c] = way_points[i][c][-1]

    return rest_locations, trajs, forces


def generate_ee_motion_trifinger(state, sol, dt_sim, dt_plan, params):
    nc = params.n_contacts
    d = params.contact_duration

    def get_simplices(surface, r=0.08):
        # hard-coded surface vertices for a cube of side length 0.625
        corners = np.array([[-r, -r], [r, -r], [-r, r], [r, r]])
        simplices = []
        for i in range(3):
            simplices.append(np.insert(corners, i, -r, axis=1))
            simplices.append(np.insert(corners, i, r, axis=1))
        return simplices[surface]

    dt_ratio = int(dt_plan / dt_sim)
    box_traj = generate_trajectories(params, dt_ratio=dt_ratio, pose_only=True)

    forces_r = []
    for c in range(nc):
        forces = [forces[0] for forces in sol.forces_world]
        forces_r.append(np.repeat(forces, repeats=100, axis=0))

    rest_locations = [[0] * nc for i in range(len(state))]
    # rest_locations[0] = [np.array([-0.05, 0, 0.08]),
    #                      np.array([0.05, 0, 0.08]),
    #                      np.array([0, 0.05, 0.08])]
    rest_locations[0] = [
        sol.locations[-1][c] + np.array([0, 0, 0.06]) for c in range(nc)
    ]

    switch = [[0] * nc for i in range(len(state))]
    trajs = [[None] * nc for i in range(len(state))]
    forces = [[None] * nc for i in range(len(state))]
    way_points = [[list() for _ in range(nc)] for i in range(len(state))]

    for i in range(len(state)):
        prev_id = i - 1 if i > 0 else i
        next_id = i + 1 if i < len(state) - 1 else i
        h = dt_ratio * d

        for c in range(nc):
            if state[i][c] == state[next_id][c]:
                if state[i][c] == 0:
                    # not in contact no switch, remains at the rest location
                    switch[i][c] = 0
                    loc = rest_locations[i][c]
                    forces[i][c] = np.zeros((h, 3))
                else:
                    # in contact no switch, follow the planned location traj
                    switch[i][c] = 1
                    loc = sol.locations[i * d][c]
                    forces[i][c] = forces_r[c][h * i : h * (i + 1)]

                way_points[i][c] += [loc, loc]
                traj_body = interpolate_trajectory_line(
                    loc, loc, d * dt_ratio - 1, dt_sim
                )

                # convert to world frame & save traj/force
                traj_ref = box_traj[h * i : h * (i + 1)]
                trajs[i][c] = location_traj_to_world(traj_body, traj_ref)

            if state[i][c] != state[next_id][c]:
                if state[next_id][c] != 0:
                    # establishing contact
                    # previous rest location -> outer cube -> inner cube -> planned location traj
                    switch[i][c] = 2
                    next_surface = state[next_id][c]
                    loc1 = rest_locations[i][c]
                    loc2 = np.mean(get_simplices(next_surface - 1), axis=0)
                    loc3 = sol.locations[next_id * d][c]
                    traj1_ref = traj2_ref = box_traj[h * i : h * (i + 1)]
                    traj1_body = interpolate_trajectory_sphere(
                        loc1, loc2, d * dt_ratio - 1, dt_sim
                    )
                    forces[i][c] = np.zeros((2 * h, 3))
                else:
                    # breaking contact
                    # complete the planned traj -> outer cube (rest location)
                    switch[i][c] = 3
                    curr_surface = state[i][c]
                    loc1 = loc2 = sol.locations[i * d][c]
                    loc3 = np.mean(get_simplices(curr_surface - 1), axis=0)

                    traj1_body = interpolate_trajectory_line(
                        loc1, loc2, d * dt_ratio - 1, dt_sim
                    )
                    traj1_ref = box_traj[h * i : h * (i + 1)]
                    ref_end = pin.XYZQUATToSE3(traj1_ref[-1])
                    traj2_ref = interpolate_trajectory(
                        ref_end, ref_end, d * dt_ratio - 1, dt_sim
                    )
                    forces[i][c] = np.vstack(
                        (forces_r[c][h * i : h * (i + 1)], np.zeros((h, 3)))
                    )

                way_points[i][c] += [loc1, loc2, loc3]
                traj2_body = interpolate_trajectory_line(
                    loc2, loc3, d * dt_ratio - 1, dt_sim
                )

                # convert to world frame
                traj1_world = location_traj_to_world(traj1_body, traj1_ref)
                traj2_world = location_traj_to_world(traj2_body, traj2_ref)

                # save traj/force
                trajs[i][c] = combine_traj(traj1_world, traj2_world)

            rest_locations[next_id][c] = way_points[i][c][-1]

    return rest_locations, trajs, forces
