import os.path as osp

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rl_utils.common import group_trajectories
from imitation_learning.utils.envs.pointmass_obstacle import PointMassObstacleEnv
from imitation_learning.utils.evaluator import Evaluator


class PMDistReward:
    def __init__(self, slack, **kwargs):
        self.slack = slack

    def __call__(self, cur_pos, prev_pos, action):
        cur_dist = torch.linalg.norm(cur_pos, dim=-1)
        prev_dist = torch.linalg.norm(prev_pos, dim=-1)
        return ((prev_dist - cur_dist) - self.slack).view(-1, 1)


class PMDistActionPenReward:
    def __init__(self, slack, action_pen, **kwargs):
        self.slack = slack
        self.action_pen = action_pen

    def __call__(self, cur_pos, prev_pos, action):
        cur_dist = torch.linalg.norm(cur_pos, dim=-1)
        prev_dist = torch.linalg.norm(prev_pos, dim=-1)

        return (
            (prev_dist - cur_dist)
            - self.slack
            - (self.action_pen * torch.linalg.norm(action, dim=-1))
        ).view(-1, 1)


class PMSparseReward:
    def __init__(self, succ_dist, **kwargs):
        self._succ_dist = succ_dist

    def __call__(self, cur_pos, prev_pos, action):
        cur_dist = torch.linalg.norm(cur_pos, dim=-1)
        reward = torch.full(cur_dist.shape, -0.1)
        reward[cur_dist < self._succ_dist] = 1.0
        return reward.view(-1, 1)


class PMSparseDenseReward:
    def __init__(self, reward_thresh, slack, **kwargs):
        self._reward_thresh = reward_thresh
        self._slack = slack

    def __call__(self, cur_pos, prev_pos, action):
        cur_dist = torch.linalg.norm(cur_pos, dim=-1)

        reward = torch.full(cur_dist.shape, -self._slack)
        assign = -self._slack * cur_dist
        should_give_reward = cur_dist < self._reward_thresh
        reward[should_give_reward] = assign[should_give_reward]
        return reward.view(-1, 1)


def viz_trajs(trajs: torch.Tensor, plt_lim, agent_point_size, fig, ax, with_arrows):
    pal = sns.color_palette("rocket", as_cmap=True)
    traj_len = trajs.size(1)

    assert len(trajs.shape) == 3
    assert trajs.shape[-1] == 2

    scatter_len = 0.1

    for i in range(trajs.size(1)):
        ax.scatter(
            trajs[:, i, 0],
            trajs[:, i, 1],
            color=[pal((i + 1) / traj_len) for _ in range(trajs.size(0))],
            s=180,
            # s=agent_point_size,
            cmap=pal,
        )
    if with_arrows:
        for i in range(trajs.size(0)):
            traj_x = trajs[i, :, 0]
            traj_y = trajs[i, :, 1]
            for t in range(trajs.size(1) - 1):
                offset = np.array(
                    [traj_x[t + 1] - traj_x[t], traj_y[t + 1] - traj_y[t]]
                )
                offset_dist = np.linalg.norm(offset)

                point_offset = offset * (scatter_len / offset_dist)
                if offset_dist < 0.05:
                    continue
                ax.arrow(
                    x=traj_x[t] + point_offset[0],
                    y=traj_y[t] + point_offset[1],
                    dx=offset[0] - (2 * point_offset[0]),
                    dy=offset[1] - (2 * point_offset[1]),
                    length_includes_head=True,
                    width=0.04,
                    head_length=0.05,
                    # color=np.array([236, 240, 241, 200]) / 255.0,
                    color=np.array([44, 62, 80]) / 255.0,
                    # color=np.array([0, 0, 0, 255]) / 255.0,
                )

    ax.set_xlim(-plt_lim, plt_lim)
    ax.set_ylim(-plt_lim, plt_lim)


def plot_obstacles(obstacle_transform, obstacle_len, obstacle_width):
    points = torch.tensor(
        [
            [-obstacle_len, -obstacle_width, 1],
            [-obstacle_len, obstacle_width, 1],
            [obstacle_len, -obstacle_width, 1],
            [obstacle_len, obstacle_width, 1],
        ]
    )

    obstacle_points = obstacle_transform @ points


class PointMassVisualizer(Evaluator):
    def __init__(
        self,
        envs,
        num_envs,
        num_steps,
        info_keys,
        rnn_hxs_dim,
        num_render,
        vid_dir,
        fps,
        save_traj_name,
        updater,
        agent_point_size,
        plt_lim,
        num_demo_plot,
        plt_density,
        plot_il,
        plot_expert: bool,
        logger,
        device,
        with_arrows: bool = False,
        is_final_render: bool = False,
        **kwargs,
    ):
        super().__init__(
            envs,
            rnn_hxs_dim,
            num_render,
            vid_dir,
            fps,
            num_envs,
            num_steps,
            info_keys,
            device,
            save_traj_name,
        )
        self._agent_point_size = agent_point_size
        self._plt_lim = plt_lim
        self._plt_density = plt_density
        self._plot_il = plot_il
        self.logger = logger
        self.device = device
        self.is_final_render = is_final_render
        self.with_arrows = with_arrows

        if plot_il and plot_expert:
            dones = updater.dataset.get_data("terminals")
            grouped_trajs = group_trajectories(dones, **updater.dataset.all_data)
            obs_trajs = (
                torch.stack([traj["observations"] for traj in grouped_trajs], dim=0)
                .detach()
                .cpu()
            )

            add_str = ""
            if num_demo_plot < obs_trajs.size(0):
                plot_idxs = torch.randint(high=len(obs_trajs), size=(num_demo_plot,))
                obs_trajs = obs_trajs[plot_idxs]
                add_str = f" (Subsampled {num_demo_plot})"

            fig, ax = plt.subplots(figsize=(4, 4))
            viz_trajs(
                obs_trajs, plt_lim, agent_point_size, fig, ax, self.is_final_render
            )
            self.plot_obstacle(ax)
            ax.set_title(f"Expert Demos{add_str}")
            self.save("demos", fig)
            plt.clf()

        self._updater = updater

    def save(self, name, fig):
        if self.is_final_render:
            full_path = osp.join(self._vid_dir, f"{name}.pdf")
            print(f"Saved to {full_path}")
            fig.savefig(full_path, bbox_inches="tight", dpi=100)
        else:
            full_path = osp.join(self._vid_dir, f"{name}.png")
            fig.savefig(full_path)
        return full_path

    def plot_obstacle(self, ax):
        if isinstance(self._envs, PointMassObstacleEnv):
            for (
                obs_xy,
                obs_width,
                obs_len,
                obs_angle,
            ) in self._envs._params.square_obstacles:
                obs_xy = [obs_xy[0] - (obs_width / 2), obs_xy[1] - (obs_len / 2)]

                color = "orangered"
                rect = Rectangle(
                    obs_xy,
                    obs_width,
                    obs_len,
                    angle=obs_angle,
                    linewidth=2,
                    edgecolor=color,
                    facecolor=color,
                )
                ax.add_patch(rect)

    def plot_reward(self, reward_fn, fig, ax):
        with torch.no_grad():
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(-self._plt_lim, self._plt_lim, self._plt_density),
                    torch.linspace(-self._plt_lim, self._plt_lim, self._plt_density),
                    indexing="ij",
                ),
                -1,
            ).to(self.device)
            reward_vals = reward_fn(cur_obs=coords, next_obs=coords).cpu()

        im_fig = ax.imshow(
            reward_vals,
            extent=[-self._plt_lim, self._plt_lim, -self._plt_lim, self._plt_lim],
            origin="lower",
        )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        self.plot_obstacle(ax)

        def fmt(x, pos):
            return str(x).ljust(5)

        fig.colorbar(
            im_fig, cax=cax, orientation="vertical", format=ticker.FuncFormatter(fmt)
        )

    def evaluate(self, policy, num_episodes, eval_i):
        fig, ax = plt.subplots(figsize=(4, 4))
        if self.is_final_render:
            ax.axis("off")
        if self._plot_il:
            self.plot_reward(
                self._updater.viz_reward,
                fig,
                ax,
            )

            self.save(f"reward_{eval_i}", fig)
            # Intentionally don't clear plot so the evaluation rollouts are
            # overlaid on reward.

        eval_result = super().evaluate(policy, num_episodes, eval_i)

        if len(self.eval_trajs_dones):
            grouped_trajs = group_trajectories(
                torch.stack(self.eval_trajs_dones, dim=0),
                obs=torch.stack(self.eval_trajs_obs, dim=0),
            )

            obs_trajs = (
                torch.stack([traj["obs"] for traj in grouped_trajs], dim=0)
                .detach()
                .cpu()
            )

            viz_trajs(
                obs_trajs,
                self._plt_lim,
                self._agent_point_size,
                fig,
                ax,
                self.with_arrows,
            )
            if not self._plot_il:
                self.plot_obstacle(ax)
            if not self.is_final_render:
                ax.set_title(
                    f"Evaluation Rollouts (Update {eval_i}, Dist {eval_result['dist_to_goal']:.4f}) "
                )
            eval_rollouts_path = self.save(f"eval_rollouts_{eval_i}", fig)
            plt.clf()

            if not self.is_final_render:
                self.logger.collect_img("reward", eval_rollouts_path)

        return eval_result
