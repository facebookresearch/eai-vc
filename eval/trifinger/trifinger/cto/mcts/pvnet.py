import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class PolicyValueNet(nn.Module):
    def __init__(self, input_size=2, action_size=31, goal_size=12, hidden_size=10):
        # game params
        super().__init__()

        self.rnn = nn.RNN(
            input_size, hidden_size, nonlinearity="relu", batch_first=True
        )

        self.fc0 = nn.Linear(goal_size, 1024)
        self.bn0 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(hidden_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)

        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)

        self.dropout = nn.Dropout(p=0.5)

        self.policy_head = nn.Linear(512, action_size)
        self.value_head = nn.Linear(512, 1)

    def forward(self, state, goal):

        output, h = self.rnn(state)
        x = h[0]

        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        g = self.dropout(F.relu(self.bn0(self.fc0(goal))))

        x = torch.cat((x, g), 1)

        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))

        pi = F.log_softmax(self.policy_head(x), dim=1)  # batch_size x action_size
        v = self.value_head(x)  # batch_size x 1

        return pi, v


class ValueClassifier(nn.Module):
    def __init__(self, input_size=2, action_size=31, goal_size=12, hidden_size=10):
        # game params
        super().__init__()

        self.rnn = nn.RNN(
            input_size, hidden_size, nonlinearity="relu", batch_first=True
        )

        self.fc0 = nn.Linear(goal_size, 1024)
        self.bn0 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(hidden_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)

        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)

        self.dropout = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, state, goal):

        output, h = self.rnn(state)
        x = h[0]

        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        g = self.dropout(F.relu(self.bn0(self.fc0(goal))))

        x = torch.cat((x, g), 1)

        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))

        y = self.fc4(x)  # batch_size x 1

        return y


def pad_collate(batch):
    (states, values, action_probs, goals) = zip(*batch)
    state_lens = torch.tensor([len(state) for state in states]).cpu()
    states = [torch.tensor(state) for state in states]
    values = torch.tensor(values)[:, None]
    action_probs = torch.tensor(np.array(action_probs))
    goals = torch.tensor(np.array(goals))
    states_pad = pad_sequence(states, batch_first=True, padding_value=0)
    return states_pad, state_lens, values, action_probs, goals
