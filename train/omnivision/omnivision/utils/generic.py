#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified from https://github.com/facebookresearch/ClassyVision/blob/main/classy_vision/generic/util.py

from dataclasses import fields, is_dataclass
from typing import Mapping, Sequence, Union

import torch


def dataclass_as_dict(obj):
    # replacement for dataclasses.asdict which makes a deepcopy of everything
    if is_dataclass(obj):
        return {f.name: dataclass_as_dict(getattr(obj, f.name)) for f in fields(obj)}
    return obj


def convert_to_one_hot(
    targets: torch.Tensor, classes, is_one_idexed=False
) -> torch.Tensor:
    """
    This function converts target class indices to one-hot vectors,
    given the number of classes.

    Set is_one_idexed, if targets are in [1, classes]
    """
    if is_one_idexed:
        targets -= 1
    assert torch.max(targets).item() < classes and torch.min(targets).item() >= 0, (
        f"Class Index [{torch.max(targets).item()}] must"
        f" be less than number of classes [{classes}]"
    )
    one_hot_targets = torch.zeros(
        (targets.shape[0], classes), dtype=torch.long, device=targets.device
    )
    one_hot_targets.scatter_(1, targets.long(), 1)
    return one_hot_targets


def convert_int_or_intlist_to_one_or_multi_hot(
    targets: Union[int, Sequence[int]], classes
) -> torch.Tensor:
    """
    Given a single integer or a list of integers (multilabel),
    return a 1D one or multihot tensor
    """
    if isinstance(targets, int):
        targets = [targets]
    targets = torch.tensor(targets).reshape((-1, 1))
    all_one_hot = convert_to_one_hot(targets, classes)
    return torch.sum(all_one_hot, dim=0)


def maybe_convert_to_one_hot(
    target: torch.Tensor, model_output: torch.Tensor
) -> torch.Tensor:
    """
    This function infers whether target is integer or 0/1 encoded
    and converts it to 0/1 encoding if necessary.
    """
    target_shape_list = list(target.size())

    if len(target_shape_list) == 1 or (
        len(target_shape_list) == 2 and target_shape_list[1] == 1
    ):
        target = convert_to_one_hot(target.view(-1, 1), model_output.shape[1])

    # target are not necessarily hard 0/1 encoding. It can be soft
    # (i.e. fractional) in some cases, such as mixup label
    assert (
        target.shape == model_output.shape
    ), "Target must of the same shape as model_output."

    return target


def is_on_gpu(model: torch.nn.Module) -> bool:
    """
    Returns True if all parameters of a model live on the GPU.
    """
    assert isinstance(model, torch.nn.Module)
    on_gpu = True
    has_params = False
    for param in model.parameters():
        has_params = True
        if not param.data.is_cuda:
            on_gpu = False
    return has_params and on_gpu


def change_dtype(tensor, dtype: str):
    if dtype == "long":
        return tensor.long()
    elif dtype == "float":
        return tensor.float()
    else:
        raise NotImplementedError()


def csv_str_to_int_tensor(csv_str: str) -> torch.IntTensor:
    """
    Convert a string of integers comma separated to a int tensor
    """
    return torch.tensor([int(el) for el in csv_str.split(",")], dtype=torch.int)


def recursive_to(data, *args, **kwargs):
    if isinstance(data, torch.Tensor):
        ret = data.to(*args, **kwargs)
    elif isinstance(data, Mapping):
        ret = type(data)()
        for key in data:
            ret[key] = recursive_to(data[key], *args, **kwargs)
    elif isinstance(data, Sequence):
        ret = type(data)()
        for value in data:
            ret.append(recursive_to(value, *args, **kwargs))
    elif is_dataclass(data):
        ret_cls = type(data)
        ret = ret_cls(**recursive_to(dataclass_as_dict(data), *args, **kwargs))
    else:
        ret = data
    return ret
