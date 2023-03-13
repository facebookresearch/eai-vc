from trifinger_envs.cube_env import ActionType
from trifinger_simulation.trifinger_platform import ObjectType
from trifinger_envs.cube_reach import CubeReachEnv


def init_reach_env():
    sim_time_step = 0.004
    downsample_time_step = 0.2
    traj_scale = 1
    n_fingers_to_move = 1
    a_dim = n_fingers_to_move * 3
    task = "reach_cube"
    state_type = "ftpos_obj"
    # obj_state_type = "mae_vit_base_patch16_ego4d_210_epochs"
    goal_type = "goal_none"

    step_size = int(downsample_time_step / sim_time_step)
    object_type = ObjectType.COLORED_CUBE
    env = CubeReachEnv(
        action_type=ActionType.TORQUE,
        step_size=step_size,
        visualization=False,
        enable_cameras=True,
        finger_type="trifingerpro",
        camera_delay_steps=0,
        time_step=sim_time_step,
        object_type=object_type,
        enable_shadows=False,
        camera_view="default",
        arena_color="default",
        visual_observation=True,
        run_rl_policy=False,
    )
    return env
