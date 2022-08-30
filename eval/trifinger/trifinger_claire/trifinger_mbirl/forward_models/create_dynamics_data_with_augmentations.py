import torch
import numpy as np
import sys
import os
from os.path import dirname, abspath

_ROOT_DIR = dirname(dirname(abspath(__file__)))
print(_ROOT_DIR)
sys.path.append(_ROOT_DIR)

data = torch.load(os.path.join(_ROOT_DIR, 'data/datasets/placing-demos-for-dynamics/demos_placing.data'))

step = 250


def noise(var, shape):
    return np.random.randn(*shape)*var

all_dyn_data ={}
for ep_name in ['move_to', 'place']:
    dyn_data = {}
    dyn_data["q_cur"] = []
    dyn_data["qd_cur"] = []
    dyn_data["keypts_cur"] = []
    dyn_data["action"] = []
    dyn_data["q_next"] = []
    dyn_data["qd_next"] = []
    dyn_data["keypts_next"] = []
    ep_data = data[ep_name]
    print(ep_name)
    print(len(ep_data))
    for r in range(len(ep_data)):
        rollout = ep_data[r]
        print(len(rollout))
        for i in range(0, len(rollout), step):
            q_cur = rollout[i]["joint_pos"]
            qd_cur = rollout[i]["joint_vel"]
            keypts_cur = np.asarray(rollout[i]["keypts"])

            inext = i+step-1

            q_next = rollout[inext]["joint_pos"]
            qd_next = rollout[inext]["joint_vel"]
            keypts_next = np.asarray(rollout[inext]["keypts"])

            dyn_data["q_cur"].append(q_cur)
            dyn_data["qd_cur"].append(qd_cur)
            dyn_data["keypts_cur"].append(keypts_cur)

            dyn_data["q_next"].append(q_next)
            dyn_data["qd_next"].append(qd_next)
            dyn_data["keypts_next"].append(keypts_next)
            dyn_data['action'].append(q_next - q_cur)

            n = 3
            for j in range(2):
                # perturb j=6 in pos
                dyn_data["q_cur"].append(q_cur)
                dyn_data["qd_cur"].append(qd_cur)
                dyn_data["keypts_cur"].append(keypts_cur)

                action_pert = dyn_data['action'][-1].copy()
                action_pert[5] -= np.random.rand(1)*0.02
                dyn_data['action'].append(action_pert)

                dyn_data["q_next"].append(q_cur + action_pert)
                dyn_data["qd_next"].append(qd_next)
                key_pts_next = dyn_data['keypts_next'][-1].copy()
                key_pts_next[:3, 1] += np.random.rand(1)*0.05
                dyn_data["keypts_next"].append(keypts_next)

                # perturb j=6 in pos
                dyn_data["q_cur"].append(q_cur)
                dyn_data["qd_cur"].append(qd_cur)
                dyn_data["keypts_cur"].append(keypts_cur)

                action_pert = dyn_data['action'][-1].copy()
                action_pert[5] += np.random.rand(1)*0.02
                dyn_data['action'].append(action_pert)

                dyn_data["q_next"].append(q_cur + action_pert)
                dyn_data["qd_next"].append(qd_next)
                key_pts_next = dyn_data['keypts_next'][-1].copy()
                key_pts_next[:3, 1] -= np.random.rand(1)*0.05
                dyn_data["keypts_next"].append(keypts_next)

            for j in range(2):
                # perturb j=6 in pos
                dyn_data["q_cur"].append(q_cur)
                dyn_data["qd_cur"].append(qd_cur)
                dyn_data["keypts_cur"].append(keypts_cur)

                action_pert = dyn_data['action'][-1].copy()
                action_pert[5] -= np.random.rand(1)*0.01
                dyn_data['action'].append(action_pert)

                dyn_data["q_next"].append(q_cur + action_pert)
                dyn_data["qd_next"].append(qd_next)
                key_pts_next = dyn_data['keypts_next'][-1].copy()
                key_pts_next[:3, 1] += np.random.rand(1)*0.03
                dyn_data["keypts_next"].append(keypts_next)

                # perturb j=6 in pos
                dyn_data["q_cur"].append(q_cur)
                dyn_data["qd_cur"].append(qd_cur)
                dyn_data["keypts_cur"].append(keypts_cur)

                action_pert = dyn_data['action'][-1].copy()
                action_pert[5] += np.random.rand(1)*0.01
                dyn_data['action'].append(action_pert)

                dyn_data["q_next"].append(q_cur + action_pert)
                dyn_data["qd_next"].append(qd_next)
                key_pts_next = dyn_data['keypts_next'][-1].copy()
                key_pts_next[:3, 1] -= np.random.rand(1)*0.03
                dyn_data["keypts_next"].append(keypts_next)

            for j in range(3):
                # perturb j=4 in pos
                dyn_data["q_cur"].append(q_cur)
                dyn_data["qd_cur"].append(qd_cur)
                dyn_data["keypts_cur"].append(keypts_cur)

                action_pert = dyn_data['action'][-1].copy()
                action_pert[3] -= np.random.rand(1)*0.02
                dyn_data['action'].append(action_pert)

                dyn_data["q_next"].append(q_cur + action_pert)
                dyn_data["qd_next"].append(qd_next)
                key_pts_next = dyn_data['keypts_next'][-1].copy()
                key_pts_next[:3, 1] += np.random.rand(1)*0.05
                dyn_data["keypts_next"].append(keypts_next)

                # perturb j=4 in pos
                dyn_data["q_cur"].append(q_cur)
                dyn_data["qd_cur"].append(qd_cur)
                dyn_data["keypts_cur"].append(keypts_cur)

                action_pert = dyn_data['action'][-1].copy()
                action_pert[3] += np.random.rand(1)*0.02
                dyn_data['action'].append(action_pert)

                dyn_data["q_next"].append(q_cur + action_pert)
                dyn_data["qd_next"].append(qd_next)
                key_pts_next = dyn_data['keypts_next'][-1].copy()
                key_pts_next[:3, 1] -= np.random.rand(1)*0.05
                dyn_data["keypts_next"].append(keypts_next)

                # perturb j=2 in pos
                dyn_data["q_cur"].append(q_cur)
                dyn_data["qd_cur"].append(qd_cur)
                dyn_data["keypts_cur"].append(keypts_cur)

                action_pert = dyn_data['action'][-1].copy()
                action_pert[1] -= np.random.rand(1)*0.02
                dyn_data['action'].append(action_pert)

                dyn_data["q_next"].append(q_cur + action_pert)
                dyn_data["qd_next"].append(qd_next)
                key_pts_next = dyn_data['keypts_next'][-1].copy()
                key_pts_next[:3, 0] += np.random.rand(1)*0.05
                dyn_data["keypts_next"].append(keypts_next)

                # perturb j=2 in pos
                dyn_data["q_cur"].append(q_cur)
                dyn_data["qd_cur"].append(qd_cur)
                dyn_data["keypts_cur"].append(keypts_cur)

                action_pert = dyn_data['action'][-1].copy()
                action_pert[1] += np.random.rand(1)*0.02
                dyn_data['action'].append(action_pert)

                dyn_data["q_next"].append(q_cur + action_pert)
                dyn_data["qd_next"].append(qd_next)
                key_pts_next = dyn_data['keypts_next'][-1].copy()
                key_pts_next[:3, 0] -= np.random.rand(1)*0.05
                dyn_data["keypts_next"].append(keypts_next)


    all_dyn_data[ep_name] = {}
    all_dyn_data[ep_name]['data'] = dyn_data

torch.save(all_dyn_data, os.path.join(_ROOT_DIR, 'data/datasets/placing-demos-for-dynamics/placing_coarse_dynamics_data_augmented.data'))