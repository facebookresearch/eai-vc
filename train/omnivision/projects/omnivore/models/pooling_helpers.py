import torch
import torch.nn as nn


class SelectElement(nn.Module):
    def __init__(self, index) -> None:
        super().__init__()
        self.index = index

    def forward(self, x):
        assert x.ndim >= 3
        return x[:, self.index, ...]


class AvgTokens(nn.Module):
    def __init__(self, start_index: int = 1) -> None:
        super().__init__()
        self.start_index = start_index

    def forward(self, x):
        assert x.ndim >= 3
        tokens = x[:, self.start_index :, ...]
        return tokens.mean(dim=1)


class SelectEOSAndProject(nn.Module):
    """
    Text Pooling used in OpenCLIP
    """

    def __init__(self, proj: nn.Module) -> None:
        super().__init__()
        self.proj = proj

    def forward(self, x, seq_len):
        assert x.ndim == 3
        # x is of shape B x L x D
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), seq_len]
        x = self.proj(x)
        return x
