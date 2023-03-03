#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import habitat
import os

from PIL import Image
from habitat.utils.visualizations.utils import (
    observations_to_image,
    images_to_video,
    append_text_to_image,
)

from habitat_vc.il.objectnav.dataset import ObjectNavDatasetV2

config = habitat.get_config("configs/tasks/objectnav_hm3d_il.yaml")


def make_videos(observations_list, output_prefix, ep_id):
    prefix = output_prefix + "_{}".format(ep_id)
    images_to_video(observations_list[0], output_dir="demos", video_name=prefix)


def save_image(img, file_name):
    im = Image.fromarray(img)
    im.save("demos/" + file_name)


def run_reference_replay(
    cfg,
    num_episodes=None,
    output_prefix=None,
    append_instruction=False,
    save_videos=False,
    save_step_image=False,
):
    possible_actions = cfg.TASK.POSSIBLE_ACTIONS
    with habitat.Env(cfg) as env:
        total_success = 0
        spl = 0

        num_episodes = min(num_episodes, len(env.episodes))
        episode_meta = []
        print("Replaying {}/{} episodes".format(num_episodes, len(env.episodes)))
        for ep_id in range(num_episodes):
            observation_list = []
            env.reset()

            step_index = 1
            total_reward = 0.0
            episode = env.current_episode

            for step_id, data in enumerate(
                env.current_episode.reference_replay[step_index:]
            ):
                action = possible_actions.index(data.action)
                action_name = env.task.get_action_name(action)

                observations = env.step(action=action)

                info = env.get_metrics()
                frame = observations_to_image({"rgb": observations["rgb"]}, info)

                if append_instruction:
                    frame = append_text_to_image(
                        frame, "Find and go to {}".format(episode.object_category)
                    )

                if save_step_image:
                    save_image(
                        frame, "trajectory_1/demo_{}_{}.png".format(ep_id, step_id)
                    )

                observation_list.append(frame)
                if action_name == "STOP":
                    break

            if save_videos:
                make_videos([observation_list], output_prefix, ep_id)
            print(
                "Total reward: {}, Success: {}, Steps: {}, Attempts: {}".format(
                    total_reward,
                    info["success"],
                    len(episode.reference_replay),
                    episode.attempts,
                )
            )

            if len(episode.reference_replay) <= 500 and episode.attempts == 1:
                total_success += info["success"]
                spl += info["spl"]

            episode_meta.append(
                {
                    "scene_id": episode.scene_id,
                    "episode_id": episode.episode_id,
                    "metrics": info,
                    "steps": len(episode.reference_replay),
                    "attempts": episode.attempts,
                    "object_category": episode.object_category,
                }
            )

        print("SPL: {}, {}, {}".format(spl / num_episodes, spl, num_episodes))
        print(
            "Success: {}, {}, {}".format(
                total_success / num_episodes, total_success, num_episodes
            )
        )

        output_path = os.path.join(
            os.path.dirname(cfg.DATASET.DATA_PATH), "replay_meta.json"
        )
        # write_json(episode_meta, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="replays/demo_1.json.gz")
    parser.add_argument("--output-prefix", type=str, default="demo")
    parser.add_argument("--num-episodes", type=int, default=10000)
    parser.add_argument(
        "--append-instruction", dest="append_instruction", action="store_true"
    )
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--save-videos", dest="save_videos", action="store_true")
    parser.add_argument(
        "--save-step-image", dest="save_step_image", action="store_true"
    )
    args = parser.parse_args()
    cfg = config
    cfg.defrost()
    cfg.DATASET.DATA_PATH = args.path
    cfg.DATASET.MAX_EPISODE_STEPS = args.max_steps
    cfg.ENVIRONMENT.MAX_EPISODE_STEPS = args.max_steps
    cfg.freeze()

    run_reference_replay(
        cfg,
        num_episodes=args.num_episodes,
        output_prefix=args.output_prefix,
        append_instruction=args.append_instruction,
        save_videos=args.save_videos,
        save_step_image=args.save_step_image,
    )


if __name__ == "__main__":
    main()
