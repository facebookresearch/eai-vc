"""
Run MoveCubePolicy to generate cube re-posititioning demos
"""

import sys
import os
import os.path
import argparse
import numpy as np

from trifinger_simulation.trifinger_platform import ObjectType
from trifinger_envs.cube_env import SimCubeEnv, ActionType
from gen_demos.reach_cube_policy import ReachCubePolicy
import utils.data_utils as d_utils
import control.cube_utils as c_utils

SIM_TIME_STEP = 0.004
FINGER_TO_MOVE = 0


def main(args):
    finger_type = "trifinger_meta"
    final_scaled_err_thresh = 0.05

    env = SimCubeEnv(
        goal_pose=None,  # passing None to sample a random trajectory
        action_type=ActionType.TORQUE,
        visualization=args.visualize,
        no_collisions=args.no_collisions,
        difficulty=args.difficulty,
        enable_cameras=True,
        finger_type=finger_type,
        time_step=SIM_TIME_STEP,
        camera_delay_steps=0,
        object_type=ObjectType.COLORED_CUBE,
        enable_shadows=False,
        camera_view="default",
        arena_color="default",
        random_q_init=args.random_q_init,
        fix_cube_base=True,
    )

    if args.log_paths:
        num_episodes = len(args.log_paths)
    else:
        num_episodes = 20

    policy = ReachCubePolicy(
        env.action_space,
        env.platform,
        time_step=SIM_TIME_STEP,
        finger_type=finger_type,
        finger_to_move=FINGER_TO_MOVE,
    )

    for i in range(num_episodes):

        final_scaled_err = np.inf

        while final_scaled_err > final_scaled_err_thresh:
            print(f"\nRunning episode {i}")

            is_done = False

            observation = env.reset(random_init_cube_pos=args.random_init_cube_pos)
            policy.reset(observation)

            observation_list = []

            while not is_done:
                action = policy.predict(observation)
                observation, reward, episode_done, info = env.step(action)

                policy_observation = policy.get_observation()

                is_done = policy.done or episode_done

                full_observation = {**observation, **policy_observation}

                # if args.log_paths is not None:
                observation_list.append(full_observation)

            # Compute scaled error for reaching task for finger_to_move
            final_ft_pos = observation_list[-1]["policy"]["controller"]["ft_pos_cur"]
            init_ft_pos = observation_list[0]["policy"]["controller"]["ft_pos_cur"]
            cube_pos = observation_list[0]["object_position"]

            final_scaled_err = d_utils.get_reach_scaled_err(
                [FINGER_TO_MOVE],
                init_ft_pos,
                final_ft_pos,
                cube_pos,
                c_utils.CUBE_HALF_SIZE,
            )
            final_pos_err = observation_list[-1]["achieved_goal_position_error"]
            print("Total episode length: ", len(observation_list))
            print("Final reach scaled error: ", final_scaled_err)

            if args.log_paths is not None:
                # Compute actions (ftpos and joint state deltas) across trajectory
                d_utils.add_actions_to_obs(observation_list)
                log_path = args.log_paths[i]
                np.savez_compressed(log_path, data=observation_list)
                print(f"Saved episode {i} to {log_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--difficulty",
        "-d",
        type=int,
        choices=[0, 1, 2, 3],
        help="Difficulty level",
        default=1,
    )
    parser.add_argument("--visualize", "-v", action="store_true", help="Visualize sim")
    parser.add_argument(
        "--no_collisions", "-nc", action="store_true", help="Visualize sim"
    )
    parser.add_argument("--log_paths", "-l", nargs="*", type=str, help="Save sim log")
    parser.add_argument(
        "--random_init_cube_pos",
        "-rp",
        action="store_true",
        help="Use random cube init pos",
    )
    parser.add_argument(
        "--random_q_init",
        "-rq",
        action="store_true",
        help="Use random init robot position",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
