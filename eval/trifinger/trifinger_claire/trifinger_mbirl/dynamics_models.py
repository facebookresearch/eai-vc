import torch
import os
import sys
import numpy as np
from scipy.optimize import approx_fprime

base_path = os.path.dirname(__file__)
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, '..'))

from envs.cube_env import SimCubeEnv, ActionType
import control.cube_utils as c_utils
from control.impedance_controller import ImpedanceController
from control.custom_pinocchio_utils import CustomPinocchioUtils

SIM_TIME_STEP = 0.004


# Compute next state given current state and action (ft position deltas)
class FTPosMPC(torch.nn.Module):

    def __init__(self, time_horizon, f_num=3):
        super().__init__()
        self.time_horizon = time_horizon
        self.f_num = f_num
        self.n_keypt_dim = self.f_num * 3
        self.a_dim = self.f_num * 3
        self.action_seq = torch.nn.Parameter(torch.Tensor(np.zeros([time_horizon, self.a_dim])))

    def forward(self, x, u=0):
        """
        Given current state and action, compute next state

        args:
            x: current state (ft_pos)
            u: action (delta ftpos)

        return:
            x_next: next state (ft_pos)
        """

        x_next = x + u
        return x_next

    def roll_out(self, obs_dict_init):
        """ Given intial state, compute trajectory of length self.time_horizon with actions self.action_seq """
        x_init = obs_dict_init["ft_state"]
        x_traj = []
        x_next = self.forward(x_init)
        x_traj.append(x_next)

        for t in range(self.time_horizon):
            a = self.action_seq[t]
            x_next = self.forward(x_next, a)
            x_traj.append(x_next.clone())

        return torch.stack(x_traj)

    def reset_actions(self):
        self.action_seq.data = torch.Tensor(np.zeros([self.time_horizon, self.a_dim]))

    def set_action_seq_for_testing(self, action_seq):
        self.action_seq.data = torch.Tensor(action_seq)


class Integrate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, action):
        ctx.save_for_backward(state, action)
        xnp = state.detach().numpy()
        unp = action.detach().numpy()
        return torch.Tensor(xnp + unp)

    @staticmethod
    def backward(ctx, grad_output):
        state, action = ctx.saved_tensors
        #grad_state = grad_action = None

        grad_state = grad_output.multiply(torch.ones_like(action))
        grad_action = grad_output.multiply(torch.ones_like(action))

        return grad_state, grad_action



# Compute next state given current state and action (ft position deltas)
class FTPosSim(torch.nn.Module):

    def __init__(self, time_horizon, f_num=3):
        super().__init__()
        self.time_horizon = time_horizon
        self.f_num = f_num
        self.n_keypt_dim = self.f_num * 3
        self.a_dim = self.f_num * 3
        self.action_seq = torch.nn.Parameter(torch.Tensor(np.zeros([time_horizon, self.a_dim])))

    def forward(self, x, u=torch.zeros(9)):
        """
        Given current state and action, compute next state

        args:
            x: current state (ft_pos)
            u: action (delta ftpos)

        return:
            x_next: next state (ft_pos)
        """

        x_next = Integrate.apply(x, u)

        return torch.Tensor(x_next)

    def roll_out(self, obs_dict_init):
        """ Given intial state, compute trajectory of length self.time_horizon with actions self.action_seq """
        x_traj = []
        x_init = obs_dict_init["ft_state"]
        x_next = self.forward(x_init)
        #x_traj.append(x_next)
        x_traj.append(torch.squeeze(x_next.clone()))

        for t in range(self.time_horizon):
            a = self.action_seq[t]
            x_next = self.forward(x_next, a)
            #x_traj.append(x_next.clone())
            x_traj.append(torch.squeeze(x_next.clone()))

        return torch.stack(x_traj)

    def reset_actions(self):
        self.action_seq.data = torch.Tensor(np.zeros([self.time_horizon, self.a_dim]))

    def set_action_seq_for_testing(self, action_seq):
        self.action_seq.data = torch.Tensor(action_seq)


class DiffTrifingerSim(torch.nn.Module):

    def __init__(self, start_pose, goal_pose, downsample_time):
        super(DiffTrifingerSim, self).__init__()
        self.env = SimCubeEnv(
            goal_pose=None,  # passing None to sample a random trajectory
            action_type=ActionType.TORQUE,
            visualization=False,
            no_collisions=True,
            enable_cameras=True,
            finger_type="trifingerpro",
            time_step=SIM_TIME_STEP,
        )

        self.observation = self.env.reset(goal_pose_dict=goal_pose, init_pose_dict=start_pose)

        self.downsample_time_step = downsample_time
        self.time_step = 0
        self.cur_state = torch.zeros(2)

        # class with kinematics functions
        self.kinematics = CustomPinocchioUtils(
            self.env.platform.simfinger.finger_urdf_path,
            self.env.platform.simfinger.tip_link_names,
            self.env.platform.simfinger.link_names)

        self.controller = ImpedanceController(self.kinematics)

    # Franzi: copied from bc_policy
    def interp_ft_traj(self, ft_pos_traj_in):
        """
        Interpolate between waypoints in ftpos trajectory, and compute velocities
        For now, just try linear interpolation between waypoints,
        with zero-order hold on linear velocity between waypoints
        """
        ft_pos_traj = c_utils.lin_interp_waypoints(ft_pos_traj_in, self.downsample_time_step,
                                                   time_step_out=self.time_step)

        ft_vel_traj = np.zeros(ft_pos_traj.shape)
        for i in range(ft_pos_traj.shape[0] - 1):
            v = (ft_pos_traj[i + 1, :] - ft_pos_traj[i, :]) / self.time_step
            ft_vel_traj[i, :] = v

        return ft_pos_traj, ft_vel_traj

    # actions are finger tip deltas
    #def forward(self, action):
    def forward(self, observation, u=torch.zeros(9)):

        ft_pos_next_des = self.ft_pos_cur + u

        # Lin interp from current ft pos to next ft waypoint
        ft_start_goal = np.stack((self.ft_pos_cur, ft_pos_next_des))
        ft_pos_traj, ft_vel_traj = self.interp_ft_traj(ft_start_goal)

        for i in range(len(ft_pos_traj)):

            # 3. Get current waypoints for finger tips
            x_des, dx_des = ft_pos_traj[i, :], ft_vel_traj[i, :] #self.get_ft_des(observation)

            #4. Get torques from controller
            q_cur = observation["robot_observation"]["position"]
            dq_cur = observation["robot_observation"]["velocity"]
            torque = self.controller.get_command_torque(x_des, dx_des, q_cur, dq_cur)
            observation, reward, episode_done, info = self.env.step(torque)

        return observation

    def roll_out(self, x_init):
        """ Given intial state, compute trajectory of length self.time_horizon with actions self.action_seq """
        x_traj = []
        x_next = self.forward(x_init)
        x_traj.append(x_next)

        for t in range(self.time_horizon):
            a = self.action_seq[t]
            x_next = self.forward(x_next, a)
            x_traj.append(x_next.clone())

        return torch.stack(x_traj)


    def get_observation(self):

        obs = {"policy":
                {
                "controller": self.controller.get_observation(),
                "t" : self.t,
                }
              }

        return obs


### Franzi: copied from another project - this shows how we cmoputed gradients via "sampling" ###
class AbstractDynamicsModel(object):
    def predict_next_state(self, state, action):
        raise NotImplementedError("Subclass must implement this function")

    def train(self, training_dataset, testing_dataset, training_params):
        raise NotImplementedError("Subclass must implement this function")

    def reset(self):
        pass

    # return Jacobian of the model with respect to input
    # the input of the model is x (state) and u (action) - concatenated.
    # the Jacobian is of size state_dim x (state_dim + action_dim)
    def dM_dxu(self, x, u):
        raise NotImplementedError("Subclass must implement this function")

    def type(self):
        raise NotImplementedError("Subclass must implement this function")


class GroundTruthForwardDynamics(AbstractDynamicsModel):

    # don't really need the torque/joint limits
    def __init__(self, model):
        model.params.gui = False
        self.dynamics_model = hydra.utils.instantiate(model)

    def __call__(self, state, u):
        return self._eval(state, u)

    def _eval(self, state, u):
        self.dynamics_model.reset_then_step(state, u)
        new_state = self.dynamics_model.get_current_joint_state()
        return new_state

    def predict_next_state(self, state, action):
        # TODO: have to decide whehter we always return mean and variance prediction, and what to do in case
        # we don't have a variance
        return self._eval(state, action), np.zeros_like(state)

    def train(self, training_dataset, testing_dataset, training_params):
        pass

    def _dx(self, x, u, eps=1e-6):
        J = np.vstack(
            [
                approx_fprime(x, lambda x: self._eval(x, u)[i], eps)
                for i in range(len(x))
            ]
        )
        return J

    def _du(self, x, u, eps=1e-6):
        J = np.vstack(
            [
                approx_fprime(u, lambda u: self._eval(x, u)[i], eps)
                for i in range(len(x))
            ]
        )
        return J

    def dM_dxu(self, x, u, eps=1e-6):
        return self.dM_dxu_finite_diff(x, u, eps)

    def dM_dxu_finite_diff(self, x, u, eps=1e-6):
        dx = self._dx(x, u, eps)
        du = self._du(x, u, eps)
        return dx, du

    def type(self):
        return "GroundTruthDynamicsModel"