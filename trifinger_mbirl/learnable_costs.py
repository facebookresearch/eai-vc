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
        mse = ((y_in[:,-self.dim:] - y_target[-self.dim:]) ** 2).squeeze()

        # weighted mse
        #wmse = torch.mm(mse, self.clip(self.weights))
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
        mse = ((y_in[:,-self.dim:] - y_target[-self.dim:]) ** 2).squeeze()
        # weighted mse
        #wmse = mse * self.clip(self.weights)
        wmse = mse * self.weights
        return wmse.mean()


class RBFWeights(torch.nn.Module):

    def __init__(self, time_horizon, dim, width, weights=None):
        super(RBFWeights, self).__init__()
        k_list = torch.linspace(0, time_horizon-1, 5) # 5 here is # of kernels (J in mbirl paper)
        if weights is None:
            self.weights = torch.nn.Parameter(0.01 * torch.ones(len(k_list), dim))
        else:
            self.weights = weights

        self.dim = dim

        x = torch.arange(0, time_horizon)
        self.K = torch.stack([torch.exp(-(int(k) - x) ** 2 / width) for k in k_list]).T
        print(f"\nRBFWEIGHTS: {k_list}")

        self.clip = torch.nn.ReLU()

    def forward(self):
        #return self.K.matmul(self.clip(self.weights))
        return self.K.matmul(self.weights)


class LearnableRBFWeightedCost(torch.nn.Module):
    def __init__(self, time_horizon, dim=9, width=2.0, weights=None):
        super(LearnableRBFWeightedCost, self).__init__()
        self.dim = dim
        self.weights_fn = RBFWeights(time_horizon=time_horizon, dim=dim, width=width, weights=weights)
        self.weights = self.weights_fn()

    def forward(self, y_in, y_target):
        assert y_in.dim() == 2
        mse = (y_in[:, -self.dim:] - y_target[-self.dim:]) ** 2

        self.weights = self.weights_fn()
        wmse = self.weights * mse

        return wmse.sum(dim=0).mean()


class BaselineCost(object):
    def __init__(self, dim, weights):
        self.weights = weights
        self.dim = dim

    def __call__(self, y_in, y_target):
        assert y_in.dim() == 2
        mse = ((y_in[:, -self.dim:] - y_target[-self.dim:]) ** 2).squeeze()

        # weighted mse
        wmse = mse * self.weights
        return wmse.mean()


