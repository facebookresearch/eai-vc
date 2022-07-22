import torch


class DeterministicPolicy(torch.nn.Module):

    def __init__(self, in_dim, out_dim, device="cpu"):
        super().__init__()
        num_neurons = 100
        self.activation = torch.nn.Tanh
        self.policy = torch.nn.Sequential(torch.nn.Linear(in_dim, num_neurons),
                                          self.activation(),
                                          torch.nn.Linear(num_neurons, num_neurons),
                                          self.activation(),
                                          torch.nn.Linear(num_neurons, out_dim),
                                          torch.nn.Tanh())
        self.policy.to(device)

        self.init_state = self.state_dict()

    def forward(self, state):
        action = self.policy(state)
        return action

    def reset(self):
        self.policy.load_state_dict(self.init_state)
