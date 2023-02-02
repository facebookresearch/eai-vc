# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import fnmatch
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import hydra
import torch
import torch.nn as nn
from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf

from .model_wrappers import MIMOHeadWrapper


def _unix_pattern_to_parameter_names(
    constraints: List[str], all_parameter_names: Set[str]
) -> Union[None, Set[str]]:
    parameter_names = []
    for param_name in constraints:
        matching_parameters = set(fnmatch.filter(all_parameter_names, param_name))
        assert (
            len(matching_parameters) > 0
        ), f"param_names {param_name} don't match any param in the given names."
        parameter_names.append(matching_parameters)
    return set.union(*parameter_names)


class CkptIncludeKernel:
    """
    Includes only the keys from the given model state_dict that match the key_pattern.
    Rest of the keys are removed from the given state_dict.

    Args:
        key_pattern: Patterns used to select the keys in the state_dict
            that are eligible for this kernel.
    """

    def __init__(self, key_pattern: List[str]):
        self.key_pattern = key_pattern

    def __call__(self, state_dict: Dict):
        """
        Args:
            state_dict: A dictionary representing the given checkpoint's state dict.
        """

        include_keys = _unix_pattern_to_parameter_names(
            self.key_pattern, state_dict.keys()
        )

        new_state_dict = {}
        for key in include_keys:
            new_state_dict[key] = state_dict[key]

        return new_state_dict


class CkptExcludeKernel:
    """
    Removes the keys from the given model state_dict that match the key_pattern.

    Args:
        key_pattern: Patterns used to select the keys in the state_dict
            that are eligible for this kernel.
    """

    def __init__(self, key_pattern: List[str]):
        self.key_pattern = key_pattern

    def __call__(self, state_dict: Dict):
        """
        Args:
            state_dict: A dictionary representing the given checkpoint's state dict.
        """

        exclude_keys = _unix_pattern_to_parameter_names(
            self.key_pattern, state_dict.keys()
        )
        include_keys = set(state_dict.keys()) - exclude_keys

        new_state_dict = {}
        for key in include_keys:
            new_state_dict[key] = state_dict[key]

        return new_state_dict


class CkptPrependKernel:
    """
    Prepends the given pattern to all the keys in the checkpoint state dict after
    selecting them with key_pattern.

    For instance, if prepend_pattern  = "some_prepend." and
    key_pattern = ["model.head"], this kernel would prepend "some_prepend." to
    "model.key", thus renaming the key "model.head" to "some_prepend.model.head".

    Args:
        prepend_pattern: The pattern to prepend the keys in the state_dict with.
        key_pattern: Patterns used to select the keys in the state_dict
            that are eligible for this kernel.
    """

    def __init__(self, prepend_pattern: str, key_pattern: Optional[List[str]] = None):
        self.prepend_pattern = prepend_pattern
        self.key_pattern = key_pattern

    def __call__(self, state_dict: Dict):
        """
        Args:
            state_dict: A dictionary representing the given checkpoint's state dict.
        """

        all_keys = set(state_dict.keys())

        include_keys = set(state_dict.keys())
        if self.key_pattern is not None:
            include_keys = _unix_pattern_to_parameter_names(
                self.key_pattern, state_dict.keys()
            )

        excluded_keys = all_keys - include_keys

        # Add excluded keys from re-mapping
        new_state_dict = {}
        for k in excluded_keys:
            new_state_dict[k] = state_dict[k]

        # Add keys from remapping
        for key in include_keys:
            new_state_dict[self.prepend_pattern + key] = state_dict[key]

        return new_state_dict


class CkptRenameKeysKernel:
    """
    Renames the key-value pairs in the checkpoint state dict. Before doing so, selects
    the keys to which to apply this kernel by using key_pattern.

    For instance, if source_pattern  = "model.head" and
    target_pattern = "model.head_1", this kernel would
    rename the key "model.head" to "model.head_1".

    Args:
        source_pattern: The pattern that needs to be renamed in the current
            checkpoint state_dict.
        target_pattern: The pattern to which the source_pattern is to be
            renamed to.
        key_pattern: Pattern used to select the keys in the state_dict
            that are eligible for this kernel. `None` selects all the keys.
    """

    def __init__(
        self,
        source_pattern: str,
        target_pattern: str,
        key_pattern: Optional[List[str]] = None,
    ):
        self.source_pattern = source_pattern
        self.target_pattern = target_pattern
        self.key_pattern = key_pattern

    def __call__(self, state_dict: Dict):
        """
        Args:
            state_dict: A dictionary representing the given checkpoint's state dict.
        """

        # Replaces only first occurences
        all_keys = set(state_dict.keys())

        include_keys = set(state_dict.keys())
        if self.key_pattern is not None:
            include_keys = _unix_pattern_to_parameter_names(
                self.key_pattern, state_dict.keys()
            )

        excluded_keys = all_keys - include_keys

        # Add excluded keys from re-mapping
        new_state_dict = {}
        for k in excluded_keys:
            new_state_dict[k] = state_dict[k]

        converted_any_key = False
        # Add keys from remapping
        for key in include_keys:
            if self.source_pattern in key:
                new_key = key.replace(self.source_pattern, self.target_pattern, 1)
                new_state_dict[new_key] = state_dict[key]
                converted_any_key = True
            else:
                new_state_dict[key] = state_dict[key]

        if not converted_any_key:
            raise KeyError(
                f"Did not find any keys with source_pattern: {self.source_pattern}"
            )

        return new_state_dict


class CkptProcessKernel:
    """
    It applies a nn.Module to keys that match the source pattern.
    This can be used to inflation etc by setting all key_patterns that
    match, for eg, trunk.module*.weight to be processed with a torch.unsqueeze
    partial function.

    Args:
        key_pattern: Patterns used to select the keys in the state_dict
            that are eligible for this kernel.
    """

    def __init__(
        self,
        key_pattern: List[str],
        processor: nn.Module,
    ):
        self.key_pattern = key_pattern
        self.processor = processor

    def __call__(self, state_dict: Dict):
        """
        Args:
            state_dict: A dictionary representing the given checkpoint's state dict.
        """
        matched_keys = list(
            _unix_pattern_to_parameter_names(self.key_pattern, state_dict.keys())
        )
        for key in matched_keys:
            state_dict[key] = self.processor(state_dict[key])
        return state_dict


def load_checkpoint(
    path_list: List[str],
    pick_recursive_keys: Optional[List[str]] = None,
    map_location: str = "cpu",
) -> Any:
    """
    Loads a checkpoint from the specified path.

    Args:
        path_list: A list of paths which contain the checkpoint. Each element
            is tried (in order) until a file that exists is found. That file is then
            used to read the checkpoint.
        pick_recursive_keys: Picks sub dicts from the loaded checkpoint if not None.
            For pick_recursive_keys = ["a", "b"], will return checkpoint_dict["a"]["b"]
        map_location (str): a function, torch.device, string or a dict specifying how to
            remap storage locations

    Returns: Model with the matchin pre-trained weights loaded.
    """
    path_exists = False
    for path in path_list:
        if g_pathmgr.exists(path):
            path_exists = True
            break

    if not path_exists:
        raise ValueError(f"No path exists in {path_list}")

    with g_pathmgr.open(path, "rb") as f:
        checkpoint = torch.load(f, map_location=map_location)

    logging.info(f"Loaded checkpoint from {path}")
    if pick_recursive_keys is not None:
        for key in pick_recursive_keys:
            checkpoint = checkpoint[key]
    return checkpoint


def get_state_dict(checkpoint, ckpt_state_dict_keys):
    pre_train_dict = checkpoint
    for i, key in enumerate(ckpt_state_dict_keys):
        if key not in pre_train_dict:
            key_str = '["' + '"]["'.join(ckpt_state_dict_keys[:i]) + '"]'
            raise KeyError(
                f"'{key}' not found in checkpoint{key_str} "
                f"with keys: {pre_train_dict.keys()}"
            )
        pre_train_dict = pre_train_dict[key]
    return pre_train_dict


def load_checkpoint_and_apply_kernels(
    checkpoint_path: str,
    checkpoint_kernels: List[Callable] = None,
    ckpt_state_dict_keys: Tuple[str] = ("state_dict",),
    map_location: str = "cpu",
) -> nn.Module:
    """
    Performs checkpoint loading with a variety of pre-processing kernel applied in
    sequence.

    Args:
        checkpoint_path (str): Path to the checkpoint.
        checkpoint_kernels List(Callable): A list of checkpoint processing kernels
            to apply in the specified order. Supported kernels include `CkptIncludeKernel`,
            `CkptExcludeKernel`, etc. These kernels are applied in the
            given order.
        ckpt_state_dict_keys (str): Keys containing the model state dict.
        map_location (str): a function, torch.device, string or a dict specifying how to
            remap storage locations

    Returns: Model with the matchin pre-trained weights loaded.
    """
    assert g_pathmgr.exists(checkpoint_path), "Checkpoint '{}' not found".format(
        checkpoint_path
    )

    # Load the checkpoint on CPU to avoid GPU mem spike.
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location=map_location)

    pre_train_dict = get_state_dict(checkpoint, ckpt_state_dict_keys)

    logging.info(
        "Loaded Checkpoint State Dict pre-kernel application: %s"
        % str(", ".join(list(pre_train_dict.keys())))
    )
    # Apply kernels
    if checkpoint_kernels is not None:
        for f in checkpoint_kernels:
            pre_train_dict = f(state_dict=pre_train_dict)

    logging.info(
        "Loaded Checkpoint State Dict Post-kernel application %s"
        % str(", ".join(list(pre_train_dict.keys())))
    )

    return pre_train_dict


def load_state_dict_into_model(state_dict: Dict, model: nn.Module, strict: bool = True):
    """
    Loads a state dict into the given model.

    Args:
        state_dict: A dictionary containing the model's
            state dict, or a subset if strict is False
        model: Model to load the checkpoint weights into
        strict: raise if the state_dict has missing state keys
    """
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    err = "State key mismatch."
    if unexpected_keys:
        err += f" Unexpected keys: {unexpected_keys}."
    if missing_keys:
        err += f" Missing keys: {missing_keys}."
    if unexpected_keys or missing_keys:
        logging.warning(err)
        if unexpected_keys or strict:
            raise KeyError(err)
    return model


def load_vissl_checkpoint(
    path_list: List[str],
    head_id_to_key_mapping: Dict[int, str],
    map_location: str = "cpu",
    strict_heads: bool = True,
    use_ema: bool = False,
    target: Type = MIMOHeadWrapper,
):
    """
    Heads are stored in an nn.Sequential in VISSL, but in an nn.ModuleDict in the
    head attacher. we use the passed mapping to do this conversion.
    Some (or all) heads can be ignored if strict_heads is False
    """
    assert target == MIMOHeadWrapper, "Only MIMOHeadWrapper is supported currently"
    checkpoint = load_checkpoint(path_list, map_location=map_location)
    model_key = "ema_model" if use_ema else "base_model"
    state = checkpoint["classy_state_dict"][model_key]["model"]
    trunk_state = state["trunk"]
    heads_state = state["heads"]
    out = {}
    for key, value in trunk_state.items():
        out[f"trunk.{key}"] = value
    for key, value in heads_state.items():
        split_idx = key.index(".")
        head_id = int(key[:split_idx])
        key_remaining = key[split_idx + 1 :]
        head_key = head_id_to_key_mapping.get(head_id)
        if head_key is None:
            if strict_heads:
                raise ValueError(f"No mapping provided for head id {head_id}")
            continue
        out[f"heads.{head_key}.{key_remaining}"] = value
    return out


def load_vissl_checkpoint_trunk_only(
    path_list: List[str],
    map_location: str = "cpu",
    use_ema: bool = False,
    model_type: str = "torch",
):
    """
    Heads are stored in an nn.Sequential in VISSL, but in an nn.ModuleDict in the
    head attacher. we use the passed mapping to do this conversion.
    Some (or all) heads can be ignored if strict_heads is False
    """

    checkpoint = load_checkpoint(path_list, map_location=map_location)

    if model_type == "torch":
        model_key = "ema_model" if use_ema else "base_model"
        state = checkpoint["classy_state_dict"][model_key]["model"]
        trunk_state = state["trunk"]
    else:
        trunk_state = checkpoint["model"]
    return trunk_state


def init_model_from_consolidated_weights(
    model, state_dict: Dict[str, Any], inflate: bool = True, ignore: bool = True
):
    # load the checkpoint now
    all_layers = model.state_dict()

    for layername in all_layers.keys():
        if layername in state_dict:
            param = state_dict[layername]
            if not isinstance(param, torch.Tensor):
                param = torch.from_numpy(param)

            # Inflate image models to video
            if inflate:
                if (
                    all_layers[layername].shape != param.shape
                    and (param.ndim == 4 or param.ndim == 5)
                    and all_layers[layername].ndim == 5
                ):
                    old_shape = param.shape
                    time_dim = all_layers[layername].size(-3)
                    if param.ndim == 4:
                        param = param.unsqueeze(-3)
                    param = param.repeat(1, 1, time_dim // param.size(2), 1, 1)
                    param = param / time_dim
                    logging.warning(
                        (f"Inflated {layername} from " f"{old_shape} to {param.shape}.")
                    )

            if all_layers[layername].shape != param.shape:
                logging.warning(
                    f"{layername} have different shapes: "
                    f"checkpoint: {param.shape}, "
                    f"model: {all_layers[layername].shape}"
                )
                if ignore:
                    continue
                else:
                    raise ValueError("Shape mismatch in checkpoint load")
            all_layers[layername].copy_(param)

    return model


def build_trunk_from_vissl_config(
    config_file_path, in_project_dir=False, model_keys=None
):
    """
    Only support generic vissl model building

    If in_project_dir is set, checks for configs in the vissl project folder.
    While doing so, loads appropriate inhererited configs using defaults.

    If in_project_dir is not set, treats config file as a single isolated file.

    model_keys (list(str)): The keys to apply in sequence on config to obtain
    the trunk config.

    #TODO: Add support for sweep files.
    """
    if in_project_dir:
        with hydra.initialize_config_module(config_module="vissl.config"):
            overrides = [f"+experiments={config_file_path}"]
            cfg = hydra.compose("defaults", overrides=overrides)
            if model_keys is None:
                model_keys = ["config", "MODEL", "TRUNK", "GENERIC"]
    else:
        with g_pathmgr.open(config_file_path, "r") as f:
            cfg = OmegaConf.load(f)
        model_keys = ["MODEL", "TRUNK", "GENERIC"]

    for k in model_keys:
        cfg = cfg[k]
    trunk = hydra.utils.instantiate(cfg)
    return trunk


def build_vit_trunk_from_vissl_config(
    config_file_path,
    in_project_dir=False,
    vit_ckpt_path=None,
    disable_final_norm=False,
    classifier="global_pool",
    drop_path=None,
    ignore_no_layers=False,
):
    """
    Builds ViT model only from vissl config while doing so,
    appropriately loads checkpoint using vissl's checkpoint loader.

    If vit_ckpt_path is given, overrides the `config.MODEL.WEIGHTS_INIT.PARAMS_FILE`
    with the given path.

    #TODO: Add support for sweep files.
    """

    import vissl.models
    from vissl.utils.checkpoint import CheckpointLoader
    from vissl.utils.hydra_config import convert_to_attrdict

    if in_project_dir:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        with hydra.initialize_config_module(config_module="vissl.config"):
            overrides = [f"+experiments={config_file_path}"]
            conf = hydra.compose("defaults", overrides=overrides)
    else:
        with g_pathmgr.open(config_file_path, "r") as f:
            cfg = OmegaConf.load(f)
            conf = OmegaConf.create()
            conf.VERSION = 1
            conf.config = cfg

    print("#################### vissl config ########################")
    print(OmegaConf.to_yaml(conf))
    print("################# vissl config end #######################")
    _, cfg = convert_to_attrdict(conf)

    if vit_ckpt_path is None:
        vit_ckpt_path = cfg["MODEL"]["WEIGHTS_INIT"]["PARAMS_FILE"]
    assert vit_ckpt_path, "Shouldn't call this when init_weight_path is empty"

    if drop_path is not None:
        cfg["MODEL"]["TRUNK"]["VISION_TRANSFORMERS"]["DROP_PATH_RATE"] = drop_path
        print(
            "Overriding Dropath to be:",
            cfg["MODEL"]["TRUNK"]["VISION_TRANSFORMERS"]["DROP_PATH_RATE"],
        )

    if ignore_no_layers:
        print("Overriding Skipped layer", cfg["MODEL"]["WEIGHTS_INIT"]["SKIP_LAYERS"])
        cfg["MODEL"]["WEIGHTS_INIT"]["SKIP_LAYERS"] = []

    model = vissl.models.build_model(cfg["MODEL"], None)

    assert CheckpointLoader.exists(vit_ckpt_path)
    checkpoint = CheckpointLoader.load_and_broadcast_init_weights(
        checkpoint_path=vit_ckpt_path, device=torch.device("cpu")
    )
    model.init_model_from_weights_params_file(cfg, checkpoint)
    trunk = model.trunk
    assert not disable_final_norm, "Manual disabling of norm is no longer supported."
    trunk.classifier_feature = classifier

    return trunk
