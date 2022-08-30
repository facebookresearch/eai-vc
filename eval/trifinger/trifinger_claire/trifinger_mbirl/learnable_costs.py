# Copyright (c) Facebook, Inc. and its affiliates.
import torch


# The learned weighted cost, with fixed weights ###
class LearnableWeightedCost(torch.nn.Module):
    def __init__(self, dim=9, weights=None):
        super(LearnableWeightedCost, self).__init__()
        if weights is None:
            self.weights = torch.nn.Parameter(0.1 * torch.ones([dim, 1]))
        else:
            self.weights = weights
        self.dim = dim
        self.clip = torch.nn.ReLU()
        self.meta_grads = [[] for _, _ in enumerate(self.parameters())]

    def forward(self, y_in, y_target):
        assert y_in.dim() == 2
        mse = (y_in - y_target) ** 2  # [time_horizon, dim]

        # weighted mse
        # wmse = torch.mm(mse, self.clip(self.weights))
        wmse = torch.mm(mse, self.weights)
        return wmse.mean()


# The learned weighted cost, with time dependent weights ###
class LearnableTimeDepWeightedCost(torch.nn.Module):
    def __init__(self, time_horizon, dim=9, weights=None):
        super(LearnableTimeDepWeightedCost, self).__init__()
        if weights is None:
            self.weights = torch.nn.Parameter(0.01 * torch.ones([time_horizon, dim]))
        else:
            self.weights = weights
        self.clip = torch.nn.ReLU()
        self.dim = dim
        self.meta_grads = [[] for _, _ in enumerate(self.parameters())]

    def forward(self, y_in, y_target):
        assert y_in.dim() == 2
        mse = ((y_in - y_target) ** 2).squeeze()
        # weighted mse
        # wmse = mse * self.clip(self.weights)
        wmse = mse * self.weights  # [time_horizon, dim]
        return wmse.mean()


class RBFWeights(torch.nn.Module):
    def __init__(self, time_horizon, dim, width, kernels=5, weights=None):
        super(RBFWeights, self).__init__()
        k_list = torch.linspace(
            0, time_horizon - 1, kernels
        )  # 5 here is # of kernels (J in mbirl paper)
        if weights is None:
            self.weights = torch.nn.Parameter(0.01 * torch.ones(len(k_list), dim))
        else:
            self.weights = weights

        self.dim = dim

        x = torch.arange(0, time_horizon)
        self.K = torch.stack(
            [torch.exp(-((int(k) - x) ** 2) / width) for k in k_list]
        ).T
        print(f"\nRBFWEIGHTS: {k_list}")

        self.clip = torch.nn.ReLU()

    def forward(self):
        # return self.K.matmul(self.clip(self.weights))
        return self.K.matmul(self.weights)


class LearnableRBFWeightedCost(torch.nn.Module):
    def __init__(self, time_horizon, dim=9, width=2.0, kernels=5, weights=None):
        super(LearnableRBFWeightedCost, self).__init__()
        self.dim = dim
        self.weights_fn = RBFWeights(
            time_horizon=time_horizon,
            dim=dim,
            width=width,
            kernels=kernels,
            weights=weights,
        )
        self.weights = self.weights_fn()

    def forward(self, y_in, y_target):
        assert y_in.dim() == 2
        mse = (y_in - y_target) ** 2

        self.weights = self.weights_fn()
        wmse = self.weights * mse

        return wmse.sum(dim=0).mean()


class LearnableFullTrajWeightedCost(torch.nn.Module):
    """Weighted MSE between full predicted trajectory and target trajectory"""

    def __init__(self, time_horizon, dim=9, weights=None):
        super(LearnableFullTrajWeightedCost, self).__init__()
        self.dim = dim
        self.weights = torch.nn.Parameter(0.1 * torch.ones([dim * time_horizon, 1]))
        # self.clip = torch.nn.ReLU()
        # self.meta_grads = [[] for _, _ in enumerate(self.parameters())]

    def forward(self, y_in, y_target):
        assert y_in.dim() == 2
        y_in = torch.flatten(y_in)
        y_target = torch.flatten(y_target)
        mse = ((y_in - y_target) ** 2).unsqueeze(0)

        # weighted mse
        # wmse = torch.mm(mse, self.clip(self.weights))
        wmse = torch.mm(mse, self.weights)
        return wmse.mean()


class LearnableMultiPhaseWeightedCost(torch.nn.Module):
    """Multi-phase cost: first, move fingertips to initial object pose, then move fingers to goal pose"""

    def __init__(self, dim=9, mode=2, weights=None):
        super(LearnableMultiPhaseWeightedCost, self).__init__()
        if weights is None:
            self.weights = torch.nn.Parameter(0.1 * torch.ones([mode, dim]))
        else:
            self.weights = weights
        self.dim = dim
        self.mode = mode

    def forward(self, y_in, y_target_per_mode):
        """
        args:
            y_in: trajectory [time_horizon, dim]
            y_target_per_mode: fingertip target positions for each phase
        """
        assert y_in.dim() == 2
        assert (
            y_target_per_mode.shape[0] == self.mode
        ), "Number of targets does not equal number of modes"

        time_horizon = y_in.shape[0]
        total_errors = torch.zeros([time_horizon, self.dim])

        for i in range(self.mode):
            sqrd_dist_to_target_i = (
                y_in - y_target_per_mode[i, :]
            ) ** 2  # [time_horizon, dim]
            total_errors += torch.mm(sqrd_dist_to_target_i, self.weights[i, :])

        return total_errors.mean()


class LearnableMultiPhaseTimeDepWeightedCost(torch.nn.Module):
    def __init__(self, time_horizon, dim=9, mode=2, weights=None):
        super(LearnableMultiPhaseTimeDepWeightedCost, self).__init__()
        if weights is None:
            self.weights = torch.nn.Parameter(1 * torch.ones([time_horizon, mode, dim]))
        else:
            self.weights = weights
        self.dim = dim
        self.mode = mode
        self.clip = torch.nn.ReLU()

    def forward(self, y_in, y_target_per_mode):
        """
        args:
            y_in: trajectory [time_horizon, dim]
        """
        assert y_in.dim() == 2
        assert (
            y_target_per_mode.shape[0] == self.mode
        ), "Number of targets does not equal number of modes"

        time_horizon = y_in.shape[0]
        total_errors = torch.zeros([time_horizon, self.dim])

        for i in range(self.mode):
            sqrd_dist_to_target_i = (
                y_in - y_target_per_mode[i, :]
            ) ** 2  # [time_horizon, dim]
            total_errors += sqrd_dist_to_target_i * self.clip(self.weights[:, i, :])

        return total_errors.mean()
