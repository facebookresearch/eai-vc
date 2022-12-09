import time
import numpy as np
from dotmap import DotMap
import pinocchio as pin
from tqdm import trange
import pybullet

from cto.envs.fingers import FingerDoubleAndObject
from cto.mcts.pvmcts import PolicyValueMCTS
from cto.miqp.problems import MIQP
from cto.trajectory import generate_random_poses
from cto.params import get_default_params, update_params


def compute_wrench_error(sol, params, scaled=True):
    total_force = np.zeros((len(sol.forces), 3))
    total_torque = np.zeros((len(sol.forces), 3))

    for n in range(len(sol.forces)):
        total_force[n] = np.sum(sol.forces[n], axis=0)
        total_torque[n] = np.sum(np.cross(sol.locations[n], sol.forces[n]), axis=0)

    force_diff = total_force - params.traj_desired.total_force
    torque_diff = total_torque - params.traj_desired.total_torque

    force_error = np.linalg.norm(force_diff, axis=1)
    torque_error = np.linalg.norm(torque_diff, axis=1)

    if scaled:
        force_error /= params.mass
        torque_error /= params.inertia[0, 0]

    return np.mean(force_error), np.mean(torque_error)


def random_poses_composite(params, n_desired_poses=2, primitive="sc"):
    z = params.box_com_height

    # task region
    lb = np.array([-0.08, -0.08, z, 0.0, 0.0, -np.pi])
    ub = np.array([0.08, 0.08, z, 0.0, 0.0, np.pi])
    # init region
    init_lb = np.array([-0.05, -0.05, z, 0.0, 0.0, -np.pi / 2])
    init_ub = np.array([0.05, 0.05, z, 0.0, 0.0, np.pi / 2])

    if primitive == "sc":
        diff_lb = np.array([-0.05, -0.05, 0, 0.0, 0.0, -np.pi / 4])
        diff_ub = np.array([0.05, 0.05, 0, 0.0, 0.0, np.pi / 4])

        desired_poses = generate_random_poses(
            n_desired_poses, lb, ub, diff_lb, diff_ub, init_lb, init_ub
        )

    elif primitive == "scl":
        diff_lb = np.array([-0.05, -0.05, 0, 0.0, 0.0, -np.pi / 4])
        diff_ub = np.array([0.05, 0.05, 0, 0.0, 0.0, np.pi / 4])

        desired_poses = generate_random_poses(
            n_desired_poses - 1, lb, ub, diff_lb, diff_ub, init_lb, init_ub
        )

        next_poses = random_poses_primitive(params, "l", desired_poses[-1])
        desired_poses.append(next_poses[-1])

    elif primitive == "scp":
        diff_lb = np.array([-0.05, -0.05, 0, 0.0, 0.0, -np.pi / 4])
        diff_ub = np.array([0.05, 0.05, 0, 0.0, 0.0, np.pi / 4])

        desired_poses = generate_random_poses(
            n_desired_poses - 1, lb, ub, diff_lb, diff_ub, init_lb, init_ub
        )

        next_poses = random_poses_primitive(params, "p", desired_poses[-1])
        desired_poses.append(next_poses[-1])

    return desired_poses


def random_poses_primitive(params, primitive, init_pose=None):
    z = params.box_com_height
    if init_pose is None:
        init_lb = np.array([-0.05, -0.05, z, 0.0, 0.0, -np.pi / 2])
        init_ub = np.array([0.05, 0.05, z, 0.0, 0.0, np.pi / 2])
    else:
        init_lb = init_pose
        init_ub = init_pose

    # task region
    lb = np.array([-0.08, -0.08, z, 0, 0, -np.pi])
    ub = np.array([0.08, 0.08, z + 0.1, 0, 0, np.pi])

    if primitive == "r":
        # rotate about the z-axis
        diff_lb = np.array([0, 0, 0, 0, 0, -np.pi / 2])
        diff_ub = np.array([0, 0, 0, 0, 0, np.pi / 2])

    elif primitive == "s":
        # slide on the xy-plane
        diff_lb = np.array([0, -0.1, 0, 0, 0, 0])
        diff_ub = np.array([0, 0.1, 0, 0, 0, 0])

    elif primitive == "sc":
        # slide on the xy-plane and rotate about the z-axis
        diff_lb = np.array([-0.05, -0.05, 0, 0, 0, -np.pi / 4])
        diff_ub = np.array([0.05, 0.05, 0, 0, 0, np.pi / 4])

    elif primitive == "p":
        # pivot about the y-axis
        random_poses = random_poses_primitive(params, "sc", init_pose)
        init_pose = random_poses[0]

        l = 0.05
        rot = np.random.uniform(0, np.pi / 4)
        th = rot + np.pi / 4
        dx = l - np.cos(th) * np.sqrt(2) * l
        dz = np.sin(th) * np.sqrt(2) * l - l

        desired_poses = [init_pose, init_pose + np.array([dx, 0, dz, 0, rot, 0])]
        return desired_poses

    elif primitive == "l":
        # lift along the z-axis
        diff_lb = np.array([0, 0, 0, 0, 0, 0])
        diff_ub = np.array([0, 0, 0.1, 0, 0, 0])

    desired_poses = generate_random_poses(1, lb, ub, diff_lb, diff_ub, init_lb, init_ub)
    return desired_poses


def create_env(params):
    pose_init = pin.SE3ToXYZQUAT(params.desired_poses[0])
    box_pos = pose_init[:3]
    box_orn = pose_init[3:]
    env = FingerDoubleAndObject(params, box_pos, box_orn, server=pybullet.DIRECT)
    return env


def exp_primitive(
    primitive,
    object_urdf,
    robot_config,
    trained_networks,
    save_results=True,
    n_trials=50,
    mcts_iter=100,
):
    # store and compute the metrics
    errors = []
    errors_miqp = []
    errors_untrained = []

    sol_time = []
    sol_time_miqp = []
    sol_time_untrained = []

    failed_tasks = []
    failed_tasks_miqp = []
    failed_tasks_untrained = []
    all_tasks = []

    for _ in trange(n_trials):
        # setup
        params = get_default_params(object_urdf, robot_config)
        params.contact_duration = 1
        desired_poses = random_poses_primitive(params, primitive)
        params = update_params(params, desired_poses)
        env = create_env(params)
        all_tasks.append(desired_poses)

        # trained mcts
        mcts = PolicyValueMCTS(params, env, networks=trained_networks)

        start = time.time()
        mcts.run(state=[[0, 0]], budget=mcts_iter, verbose=False)
        te = time.time() - start
        best_state, sol = mcts.get_solution()
        sol_time.append(te)

        if best_state is not None:
            errors.append(compute_wrench_error(sol, params))
        else:
            failed_tasks.append(desired_poses)

        ## untrained mcts
        mcts_untrained = PolicyValueMCTS(params, env)

        start = time.time()
        mcts_untrained.run(state=[[0, 0]], budget=mcts_iter, verbose=False)
        te = time.time() - start
        best_state, sol = mcts_untrained.get_solution()
        sol_time_untrained.append(te)

        if best_state is not None:
            errors_untrained.append(compute_wrench_error(sol, params))
        else:
            failed_tasks_untrained.append(desired_poses)

        ## miqp
        params = get_default_params(object_urdf, robot_config, MIQP=True)
        params = update_params(params, desired_poses)
        miqp = MIQP(params)
        miqp.setup()

        start = time.time()
        sol = miqp.solve()
        te_miqp = time.time() - start
        sol_time_miqp.append(te_miqp)
        if sol is not None:
            errors_miqp.append(compute_wrench_error(sol, params))
        else:
            failed_tasks_miqp.append(desired_poses)

        env.close()

    results = DotMap()
    results.primitive = primitive
    results.n_trials = n_trials
    results.errors = errors
    results.errors_miqp = errors_miqp
    results.errors_untrained = errors_untrained

    results.sol_time = sol_time
    results.sol_time_miqp = sol_time_miqp
    results.sol_time_untrained = sol_time_untrained

    results.failed_tasks = failed_tasks
    results.failed_tasks_miqp = failed_tasks_miqp
    results.failed_tasks_untrained = failed_tasks_untrained

    results.all_tasks = all_tasks

    if save_results:
        import pickle

        with open("logs/primitive_" + primitive + ".pkl", "wb") as f:
            pickle.dump(results, f)

    return results


def exp_composite(
    primitive,
    n_desired_poses,
    object_urdf,
    robot_config,
    trained_networks,
    save_results=True,
    n_trials=50,
    mcts_iter=200,
    include_miqp=True,
):
    # store and compute the metrics
    errors = []
    errors_miqp = []
    errors_untrained = []

    sol_time = []
    sol_time_miqp = []
    sol_time_untrained = []

    failed_tasks = []
    failed_tasks_miqp = []
    failed_tasks_untrained = []
    all_tasks = []

    for _ in trange(n_trials):
        # setup
        params = get_default_params(object_urdf, robot_config)
        params.contact_duration = 3
        desired_poses = random_poses_composite(params, n_desired_poses)
        params = update_params(params, desired_poses)
        env = create_env(params)
        all_tasks.append(desired_poses)

        # trained mcts
        mcts = PolicyValueMCTS(params, env, networks=trained_networks)

        start = time.time()
        mcts.run(state=[[0, 0]], budget=mcts_iter, verbose=False)
        te = time.time() - start
        best_state, sol = mcts.get_solution()
        sol_time.append(te)

        if best_state is not None:
            errors.append(compute_wrench_error(sol, params))
        else:
            failed_tasks.append(desired_poses)

        ## untrained mcts
        mcts_untrained = PolicyValueMCTS(params, env)

        start = time.time()
        mcts_untrained.run(state=[[0, 0]], budget=mcts_iter, verbose=False)
        te = time.time() - start
        best_state, sol = mcts_untrained.get_solution()
        sol_time_untrained.append(te)

        if best_state is not None:
            errors_untrained.append(compute_wrench_error(sol, params))
        else:
            failed_tasks_untrained.append(desired_poses)

        ## miqp
        if include_miqp:
            params = get_default_params(object_urdf, robot_config, MIQP=True)
            params = update_params(params, desired_poses)
            miqp = MIQP(params)
            miqp.setup()

            start = time.time()
            sol = miqp.solve()
            te_miqp = time.time() - start
            sol_time_miqp.append(te_miqp)
            if sol is not None:
                errors_miqp.append(compute_wrench_error(sol, params))
            else:
                failed_tasks_miqp.append(desired_poses)

        env.close()

    results = DotMap()
    results.primitive = primitive
    results.n_trials = n_trials
    results.errors = errors
    results.errors_miqp = errors_miqp
    results.errors_untrained = errors_untrained

    results.sol_time = sol_time
    results.sol_time_miqp = sol_time_miqp
    results.sol_time_untrained = sol_time_untrained

    results.failed_tasks = failed_tasks
    results.failed_tasks_miqp = failed_tasks_miqp
    results.failed_tasks_untrained = failed_tasks_untrained

    results.all_tasks = all_tasks

    if save_results:
        import pickle

        with open("logs/composite_" + primitive + ".pkl", "wb") as f:
            pickle.dump(results, f)

    return results


def print_metrics(r):
    np.set_printoptions(precision=2, suppress=True)
    print("________________________________")
    print("miqp model results")
    print("________________________________")
    try:
        print("success rate:", len(r.errors_miqp) / r.n_trials)
        print("mean computation time:", np.mean(r.sol_time_miqp))
        print("worst computation time:", np.max(r.sol_time_miqp))
        print("avg error:", np.mean(r.errors_miqp, axis=0))
        print("worst error", np.max(r.errors_miqp, axis=0))
    except:
        print("all tasks failed")

    print("________________________________")
    print("trained model results")
    print("________________________________")
    print("success rate:", len(r.errors) / r.n_trials)
    print("mean computation time:", np.mean(r.sol_time))
    print("worst computation time:", np.max(r.sol_time))
    print("avg error:", np.mean(r.errors, axis=0))
    print("worst error", np.max(r.errors, axis=0))

    print("________________________________")
    print("untrained model results")
    print("________________________________")
    print("success rate:", len(r.errors_untrained) / r.n_trials)
    print("mean computation time:", np.mean(r.sol_time_untrained))
    print("worst computation time:", np.max(r.sol_time_untrained))
    print("avg error:", np.mean(r.errors_untrained, axis=0))
    print("worst error", np.max(r.errors_untrained, axis=0))
