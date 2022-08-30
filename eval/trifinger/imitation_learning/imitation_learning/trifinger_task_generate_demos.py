import os
import torch
import numpy as np
import hydra
from hydra.utils import instantiate as hydra_instantiate
from typing import Dict
from collections import defaultdict

from rl_helper.envs import create_vectorized_envs
import imitation_learning.common.trifinger_envs as trifinger_envs
import imitation_learning
from imitation_learning.policy_opt.policy import Policy


@hydra.main(config_path="config/meta_irl", config_name="trifinger")
def main(cfg) -> Dict[str, float]:

    num_envs = 1
    cfg.num_envs = 1

    def create_trifinger_env(seed):
        np.random.seed(seed)
        env = trifinger_envs.CausalWorldReacherWrapper(
            start_state_noise=0.01, skip_frame=10, max_ep_horizon=5
        )
        return env

    envs = create_vectorized_envs(
        env_id="TriFingerReaching-v0",
        num_envs=num_envs,
        seed=cfg.seed,
        create_env_fn=create_trifinger_env,
    )

    device = torch.device("cpu")
    cfg.obs_shape = envs.observation_space.shape
    cfg.action_dim = envs.action_space.shape[0]
    cfg.action_is_discrete = False

    policy: Policy = hydra_instantiate(cfg.policy)

    ckpt = torch.load("data/trained_experts/sgd/ckpt.3124.pth")
    policy.load_state_dict(ckpt["policy"])

    all_obs = []
    all_actions = []
    all_dones = []
    all_rewards = []
    obs = envs.reset()
    all_obs.append(obs)
    rnn_hxs = torch.zeros(num_envs, 0).to(device)
    eval_masks = torch.zeros(num_envs, 1, device=device)
    for n in range(5):
        act_data = policy.act(obs, rnn_hxs, eval_masks, deterministic=True)
        next_obs, rewards, done, infos = envs.step(act_data["action"].detach())
        rnn_hxs = act_data["recurrent_hidden_states"]

        obs = next_obs
        all_obs.append(obs)
        all_actions.append(act_data["action"])
        all_dones.append(torch.Tensor([done]))
        all_rewards.append(rewards)

    dist_to_goal = [
        np.mean((x["desired_goal"] - x["achieved_goal"]) ** 2) for x in infos
    ]
    success = np.array(np.array(dist_to_goal) < 0.001, dtype=float)

    print(f"final dist to goal: {dist_to_goal}")
    print("")
    torch.set_printoptions(precision=3, sci_mode=False)
    print("joint sequence")
    print(torch.stack(all_obs).squeeze()[:, :9])
    print("")
    print("action sequence")
    print(torch.stack(all_actions))
    print("")
    print("EE sequence")
    print(torch.stack(all_obs).squeeze()[:, 9:])
    print("")

    demo_data = {}
    demo_data["observations"] = torch.stack(all_obs).squeeze()[:-1, :]
    demo_data["actions"] = torch.stack(all_actions).squeeze()
    demo_data["terminals"] = torch.stack(all_dones).squeeze()
    demo_data["rewards"] = torch.stack(all_rewards).squeeze()
    demo_data["next_observations"] = torch.stack(all_obs).squeeze()[1:, :]

    lib_dir_name = os.path.dirname(imitation_learning.__path__[0])
    torch.save(
        demo_data,
        os.path.join(
            lib_dir_name,
            "imitation_learning/data/trifinger_new_demo_sgd_expert_with_next.pt",
        ),
    )


if __name__ == "__main__":
    main()
