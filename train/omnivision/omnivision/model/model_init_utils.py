from typing import Callable, Dict

import torch.nn as nn


def init_parameters(model: nn.Module, init_fns: Dict[str, Callable]) -> nn.Module:
    for param_name, init_fn in init_fns.items():
        param = model.get_parameter(param_name)
        ret = init_fn(param)
        assert ret is None or ret is param, "init_fn should update param in place"
    return model
