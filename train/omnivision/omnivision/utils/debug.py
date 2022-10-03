import traceback
from dataclasses import dataclass, is_dataclass
from typing import Mapping, Sequence

import torch
from omnivision.utils.generic import dataclass_as_dict


@dataclass
class Tensor:
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device

    def __init__(self, tensor: torch.Tensor) -> None:
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.device = tensor.device


def get_traceback(limit=5):
    return "".join(traceback.format_stack(limit=limit))


def replace_tensor_with_metadata(data):
    if isinstance(data, torch.Tensor):
        ret = Tensor(data)
    elif isinstance(data, Mapping):
        ret = {}
        for key in data:
            ret[key] = replace_tensor_with_metadata(data[key])
    elif isinstance(data, Sequence):
        ret = []
        for value in data:
            ret.append(replace_tensor_with_metadata(value))
    elif is_dataclass(data):
        ret_cls = type(data)
        ret = ret_cls(**replace_tensor_with_metadata(dataclass_as_dict(data)))
    else:
        ret = data
    return ret
