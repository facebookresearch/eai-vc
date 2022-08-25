from typing import List

import torch.nn as nn


def make_mlp_layers(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    n_layers: int,
    activations_cls=None,
) -> List[nn.Module]:
    """
    :returns: A list of modules in the mlp.
    """
    if activations_cls is None:
        activations_cls = nn.Tanh
    net_layers = [nn.Linear(input_dim, hidden_dim)]
    for _ in range(n_layers):
        net_layers.extend(
            [
                activations_cls(),
                nn.Linear(hidden_dim, hidden_dim),
            ]
        )
    net_layers.extend([activations_cls(), nn.Linear(hidden_dim, output_dim)])
    return net_layers
