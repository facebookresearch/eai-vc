#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python
import argparse
import glob
import multiprocessing
import os

import habitat_sim
import numpy as np
from PIL import Image

# suppress logging from habitat sim
os.environ["GLOG_minloglevel"] = "2"

VERSION = "v1"
SENSOR_RESOLUTION = 512
SENSOR_HEIGHT = 1.25
AGENT_HEIGHT = 1.5
AGENT_RADIUS = 0.1
STEP_SIZE = 0.25
TURN_ANGLE = 30
NUM_RETRIES = 100
LONG_DISTANCE = 6.0
SHORT_DISTANCE = 4.0
MIN_STEPS = 16
MAX_STEPS = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["hm3d", "gibson", "all"],
        help="dataset (default: all)",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        default=1500,
        type=int,
        help="approximate number of samples per environment (default: 3,000)",
    )
    parser.add_argument(
        "-s",
        "--split",
        default="train",
        choices=["train", "val"],
        help="dataset split (default: train)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=8,
        type=int,
        help="Number of workers (default: 8)",
    )
    args = parser.parse_args()

    args.scene_directory = "data/scene_datasets/"
    dataset_name = "hm3d+gibson" if args.dataset == "all" else args.dataset
    args.output_directory = os.path.join(
        "tmae", "data", "datasets", dataset_name, VERSION, args.split
    )

    return args


def get_scenes(args):
    scenes = []
    if args.dataset == "hm3d" or args.dataset == "all":
        folder = os.path.join(args.scene_directory, "hm3d", args.split)
        scenes += sorted(glob.glob(os.path.join(folder, "*", "*.basis.glb")))
    if args.dataset == "gibson" or args.dataset == "all":
        folder = os.path.join(args.scene_directory, "gibson")
        scenes += [os.path.join(folder, s + ".glb") for s in gibson(args.split)]
    assert all(os.path.exists(s) for s in scenes)
    return scenes


def make_cfg(scene_id):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_id

    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "rgb"
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.resolution = [SENSOR_RESOLUTION, SENSOR_RESOLUTION]
    sensor_spec.position = [0.0, SENSOR_HEIGHT, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = AGENT_HEIGHT
    agent_cfg.radius = AGENT_RADIUS
    agent_cfg.action_space["move_forward"] = habitat_sim.ActionSpec(
        "move_forward", habitat_sim.ActuationSpec(STEP_SIZE)
    )
    agent_cfg.action_space["turn_left"] = habitat_sim.ActionSpec(
        "turn_left", habitat_sim.ActuationSpec(TURN_ANGLE)
    )
    agent_cfg.action_space["turn_right"] = habitat_sim.ActionSpec(
        "turn_right", habitat_sim.ActuationSpec(TURN_ANGLE)
    )
    agent_cfg.sensor_specifications = [sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def sample_random_path(sim, min_distance):
    src = sim.pathfinder.get_random_navigable_point()
    for _ in range(NUM_RETRIES):
        tgt = sim.pathfinder.get_random_navigable_point()
        path = habitat_sim.ShortestPath()
        path.requested_start = src
        path.requested_end = tgt
        if not sim.pathfinder.find_path(path):
            continue
        if path.geodesic_distance < min_distance:
            continue
        path.requested_start = path.points[-2]
        if not sim.pathfinder.find_path(path):
            continue
        if path.geodesic_distance >= STEP_SIZE:
            continue
        return src, tgt
    return None, None


def sample_random_rotation():
    angle = np.random.uniform(-np.pi, np.pi)
    return [0.0, np.sin(angle / 2), 0.0, np.cos(angle / 2)]


def scene_id_to_scene(scene_id):
    if "hm3d" in scene_id:
        return os.path.basename(os.path.dirname(scene_id))
    return os.path.basename(scene_id).replace(".basis", "").replace(".glb", "")


def collect_data(inputs):
    scene_id, args = inputs

    # make output folder
    scene = scene_id_to_scene(scene_id)
    output_folder = os.path.join(args.output_directory, scene)
    os.makedirs(output_folder, exist_ok=True)

    # check output folder
    image_count = len(glob.glob(os.path.join(output_folder, "*", "*.jpg")))
    if image_count >= args.num_samples:
        message("skipping {}".format(scene_id))
        return

    # make simulator
    cfg = make_cfg(scene_id)
    sim = habitat_sim.Simulator(cfg)
    follower = sim.make_greedy_follower(agent_id=0, goal_radius=STEP_SIZE)

    # collect samples
    path_count = len(glob.glob(os.path.join(output_folder, "*")))
    while image_count < args.num_samples:
        # make folder
        folder = os.path.join(output_folder, f"{path_count:04d}")
        os.makedirs(folder, exist_ok=True)

        # sample path
        src, tgt = sample_random_path(sim, LONG_DISTANCE)
        if src is None or tgt is None:
            src, tgt = sample_random_path(sim, SHORT_DISTANCE)
        if src is None or tgt is None:
            continue
        rot = sample_random_rotation()

        # initialize agent
        agent = sim.get_agent(0)
        state = agent.get_state()
        state.position = src
        state.rotation = rot
        agent.set_state(state)

        # follow path
        follower.reset()
        step_count, images = 0, []
        while True:
            try:
                action = follower.next_action_along(tgt)
            except habitat_sim.errors.GreedyFollowerError:
                break
            if action is None:
                break
            images.append(sim.step(action)["rgb"])
            step_count += 1
            if step_count == MAX_STEPS:
                break

        if step_count < MIN_STEPS:
            continue

        for img in images:
            path = os.path.join(folder, f"{image_count:04d}.jpg")
            Image.fromarray(img).convert("RGB").save(path)
            image_count += 1

        path_count += 1

    message(
        f"done with {scene_id}",
        f"collected {image_count} images",
        f"from {path_count} paths",
    )


def main():
    args = parse_args()

    scenes = get_scenes(args)
    print(f"number of scenes: {len(scenes)}")

    inputs = [(scene, args) for scene in scenes]
    with multiprocessing.Pool(args.workers) as pool:
        for _ in pool.imap_unordered(collect_data, inputs):
            pass


def message(*msg):
    print("***\n" + " ".join(msg) + "\n***")


# fmt: off
def gibson(split):
    if split == "train":
        return [
            "Adrian", "Albertville", "Anaheim", "Andover", "Angiola", "Annawan",
            "Applewold", "Arkansaw", "Avonia", "Azusa", "Ballou", "Beach", "Bolton",
            "Bowlus", "Brevort", "Capistrano", "Colebrook", "Convoy", "Cooperstown",
            "Crandon", "Delton", "Dryville", "Dunmor", "Eagerville", "Goffs",
            "Hainesburg", "Hambleton", "Haxtun", "Hillsdale", "Hometown", "Hominy",
            "Kerrtown", "Maryhill", "Mesic", "Micanopy", "Mifflintown", "Mobridge",
            "Monson", "Mosinee", "Nemacolin", "Nicut", "Nimmons", "Nuevo", "Oyens",
            "Parole", "Pettigrew", "Placida", "Pleasant", "Quantico", "Rancocas",
            "Reyno", "Roane", "Roeville", "Rosser", "Roxboro", "Sanctuary",
            "Sasakwa", "Sawpit", "Seward", "Shelbiana", "Silas", "Sodaville",
            "Soldier", "Spencerville", "Spotswood", "Springhill", "Stanleyville",
            "Stilwell", "Stokes", "Sumas", "Superior", "Woonsocket",
        ]
    elif split == "val":
        return [
            "Cantwell", "Denmark", "Eastville", "Edgemere", "Elmira", "Eudora",
            "Greigsville", "Mosquito", "Pablo", "Ribera", "Sands", "Scioto",
            "Sisters", "Swormville",
        ]
    else:
        raise ValueError("invalid split: {}".format(split))
# fmt: on


if __name__ == "__main__":
    main()
