import contextlib
import dataclasses
import enum
import os

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import hydra

import torch
import torch.distributed as dist
import torch.nn as nn
from fairscale.nn import enable_wrap, FullyShardedDataParallel, wrap
from fairscale.optim.grad_scaler import ShardedGradScaler
from iopath.common.file_io import g_pathmgr

from omnivision.model.checkpoint_utils import get_state_dict, load_checkpoint
from omnivore.train_utils import get_machine_local_and_dist_rank, makedir


class ComputeType(enum.Enum):
    """
    The compute types available for FSDP models
    """

    FP32 = "float32"
    FP16 = "float16"
    BF16 = "bfloat16"

    def to_torch_dtype(self):
        if self == ComputeType.FP32:
            return torch.float32
        elif self == ComputeType.FP16:
            return torch.float16
        else:
            assert self == ComputeType.BF16
            return torch.bfloat16

    @classmethod
    def from_string(cls, text: str) -> "ComputeType":
        if text == cls.FP32.value:
            return cls.FP32
        elif text == cls.FP16.value:
            return cls.FP16
        elif text == cls.BF16.value:
            return cls.BF16
        else:
            options = [cls.FP32.value, cls.FP16.value, cls.BF16.value]
            msg = f"Unsupported compute type {text} for FSDP. Please use one of {options}."
            raise ValueError(msg)


@dataclass
class FSDPSettings:
    """
    Data class containing FSDP parameters used to parameterize
    either FairScale or PyTorch (not yet supported) FSDP
    """

    flatten_parameters: bool = True
    move_params_to_cpu: bool = False
    bucket_cap_mb: int = 0
    compute_dtype: str = "float32"
    mixed_precision: bool = False
    fp32_reduce_scatter: bool = False
    reshard_after_forward: bool = False
    full_precision_layers: List[str] = field(default_factory=list)

    # TODO - benchmark if this option is still useful for speed
    force_global_group: bool = True

    def to_fairscale_fsdp_settings(self) -> Dict[str, Any]:
        fsdp_config = {
            "flatten_parameters": self.flatten_parameters,
            "move_params_to_cpu": self.move_params_to_cpu,
            "bucket_cap_mb": self.bucket_cap_mb,
            "mixed_precision": self.mixed_precision,
            "fp32_reduce_scatter": self.fp32_reduce_scatter,
            "reshard_after_forward": self.reshard_after_forward,
            "compute_dtype": ComputeType.from_string(
                self.compute_dtype
            ).to_torch_dtype(),
        }
        if self.force_global_group:
            fsdp_config["process_group"] = get_global_group()
        return fsdp_config


@contextlib.contextmanager
def enable_fsdp(settings: FSDPSettings):
    """
    Context Manager used to set up FSDP parameters and enable the following functions:
    - fairscale.nn.wrap
    - specific wrapping of FP32 layers
    """
    fsdp_kwargs = settings.to_fairscale_fsdp_settings()
    is_mixed_precision = (
        settings.compute_dtype != ComputeType.FP32 or settings.mixed_precision
    )
    with enable_wrap(wrapper_cls=FullyShardedDataParallel, **fsdp_kwargs):
        # If the outer FSDP context is not in full precision, establish another
        # context to wrap the layers listed in `full_precision_layers` in FP32
        with FullPrecisionFSDPContext(
            fp32_layers=settings.full_precision_layers, enabled=is_mixed_precision
        ):
            yield


@dataclass
class _FullPrecisionFSDPSettings:
    enabled: bool = False
    fp32_layers: Tuple[Any, ...] = field(default_factory=tuple)


class FullPrecisionFSDPContext:
    """
    Context manager used to set specific FP32 precision
    to a specified list of layer types
    """

    SCOPE: _FullPrecisionFSDPSettings = _FullPrecisionFSDPSettings()

    def __init__(self, fp32_layers: List[str], enabled: bool):
        self.fp32_layers = tuple(
            self._type_from_name(layer_class_name) for layer_class_name in fp32_layers
        )
        self.enabled = enabled
        self.prev_scope = _FullPrecisionFSDPSettings()

    def __enter__(self) -> None:
        self.prev_scope = FullPrecisionFSDPContext.SCOPE
        FullPrecisionFSDPContext.SCOPE = _FullPrecisionFSDPSettings(
            fp32_layers=self.fp32_layers,
            enabled=self.enabled,
        )

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        FullPrecisionFSDPContext.SCOPE = self.prev_scope

    @staticmethod
    def _type_from_name(layer_class_name: str):
        return hydra.utils.get_class(layer_class_name)


def wrap_fsdp_if_enabled(module, **wrap_overrides: Any):
    """
    Simple wrapper around FSDP, which allows us to plug specific behavior
    for specific modules via configuration.
    """
    if isinstance(module, FullPrecisionFSDPContext.SCOPE.fp32_layers):
        if not FullPrecisionFSDPContext.SCOPE.enabled:
            return module
        wrap_overrides.update(
            {
                "compute_dtype": torch.float32,
                "mixed_precision": False,
                "fp32_reduce_scatter": None,
                "force_input_to_fp32": True,
            }
        )
    return wrap(module, **wrap_overrides)


class FSDPWrapMetaclass(type):
    """Metaclass to optionally wrap a model with FSDP if fsdp_settings are passed."""

    def __call__(cls, *args, fsdp_settings: Optional[FSDPSettings] = None, **kwargs):
        if fsdp_settings is None:
            model = super().__call__(*args, fsdp_settings=fsdp_settings, **kwargs)
        else:
            with enable_fsdp(fsdp_settings):
                model = super().__call__(*args, fsdp_settings=fsdp_settings, **kwargs)
                model = wrap_fsdp_if_enabled(model)
        return model


def get_global_group():
    """
    Singleton pytorch distributed group
    Inspired by https://github.com/pytorch/fairseq
    """
    if dist.is_initialized():
        if not hasattr(get_global_group, "_global_group"):
            # ideally we could use torch.distributed.group.WORLD, but it seems
            # to cause random NCCL hangs in some cases
            get_global_group._global_group = dist.new_group()
        return get_global_group._global_group
    else:
        return None


def is_fsdp(module: nn.Module) -> bool:
    """
    Helper function to check if a module is actually
    a FSDP wrapper module
    """
    return isinstance(module, FullyShardedDataParallel)


def is_valid_fsdp_model(model: FullyShardedDataParallel) -> bool:
    """
    Checks if a FSDP model is valid by looking at the sub-FSDP modules
    and ensuring that they do not think they are the root FSDP model
    """
    for name, module in model.named_modules():
        if is_fsdp(module):
            if name != "" and module._is_root is not None:
                return False
    return True


def setup_fsdp_distributed_training(fsdp_settings, model):
    """
    Wrap the model for distributed training and make sure
    that the model is correctly built
    """
    fsdp_settings = hydra.utils.instantiate(fsdp_settings)
    with enable_fsdp(fsdp_settings):
        model = wrap_fsdp_if_enabled(model)
        if not is_valid_fsdp_model(model):
            raise RuntimeError("The FSDP model is not correctly initialized")
        return model


def create_fsdp_grad_scaler(enabled: bool) -> ShardedGradScaler:
    """
    Simple wrapper around the ShardedGradScaler of FairScale to select
    the right default options and hide fairscale from the outside
    """
    return ShardedGradScaler(enabled=enabled, process_group=get_global_group())


def clip_fsdp_gradients(model: nn.Module, max_norm: float, norm_type: int):
    """
    Helper function to clip the gradients of a FSDP model
    """
    assert is_fsdp(model), "Make sure you are using FSDP in your model"
    model.clip_grad_norm_(max_norm=max_norm, norm_type=norm_type)


class ShardedCheckpointType(enum.Enum):
    """
    Types of checkpoints save-able and loadable by a FSDP model:
    - CONSOLIDATED: a self contained checkpoint with all the parameters
        of the model
    - SHARD: a shard of a model as saved by a worker which contains
        the parameters the worker works on, the optimizer state
        specific to this worker, etc
    - SHARD_LIST: a special type of checkpoint which does not contain
        any parameters but instead lists all the shards saved by each
        individual worker
    """

    CONSOLIDATED = 0
    SHARD = 1
    SHARD_LIST = 2


class ShardedCheckpointSaver:
    """
    Helper class to save checkpoints of FSDP models
    """

    def __init__(
        self,
        checkpoint_folder: str,
        shard_sub_folder: str = "shards",
        save_freq: int = -1,
    ):
        super().__init__()
        self.checkpoint_folder = checkpoint_folder
        self.shard_sub_folder = shard_sub_folder
        self.save_freq = save_freq
        self.world_size = torch.distributed.get_world_size()
        _, disk_rank = get_machine_local_and_dist_rank()
        self.worker_id = disk_rank

    def save_checkpoint(
        self,
        model: FullyShardedDataParallel,
        optimizer: Any,
        loss: Any,
        epoch: int,
        steps: Any,
        scaler: Any,
        train_dataset_state: Optional[Any] = None,
    ) -> None:
        assert isinstance(
            scaler, ShardedGradScaler
        ), f"Invalid type for scaler: {type(scaler)}"
        local_checkpoint = {
            "type": ShardedCheckpointType.SHARD.name,
            "model": model.local_state_dict(),
            "model_meta": model.local_metadata_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss.state_dict(),
            "epoch": epoch,
            "steps": steps,
        }
        if scaler.is_enabled():
            local_checkpoint["scaler"] = scaler.state_dict()
        # TODO - add support for EMA model
        if train_dataset_state is not None:
            local_checkpoint["train_dataset_state"] = train_dataset_state

        # Prepare the folders to save the checkpoints into
        makedir(self.checkpoint_folder)
        makedir(os.path.join(self.checkpoint_folder, self.shard_sub_folder))

        # Save the local checkpoint (on all ranks)
        paths = [
            self._get_shard_file_path(
                epoch=epoch, shard_id=self.worker_id, with_epoch=False
            ),
        ]
        if self._should_save_epoch_specific(epoch):
            paths.append(
                self._get_shard_file_path(
                    epoch=epoch, shard_id=self.worker_id, with_epoch=True
                ),
            )
        for path in paths:
            with g_pathmgr.open(path, "wb") as f:
                torch.save(local_checkpoint, f)

        # Saving the global checkpoint (only on worker 0)
        if self.worker_id == 0:
            self._save_shard_list(int(epoch))

    def _save_shard_list(self, epoch: int):
        training_checkpoint_path = os.path.join(self.checkpoint_folder, "checkpoint.pt")
        with g_pathmgr.open(training_checkpoint_path, "wb") as f:
            global_checkpoint = {
                "type": ShardedCheckpointType.SHARD_LIST.name,
                "shards_root": self.checkpoint_folder,
                "shards": [
                    self._get_shard_file_path(
                        epoch, shard_id, with_epoch=False, rel_path=True
                    )
                    for shard_id in range(self.world_size)
                ],
            }
            torch.save(global_checkpoint, f)

        if self._should_save_epoch_specific(epoch):
            checkpoint_path = os.path.join(
                self.checkpoint_folder, f"checkpoint_{epoch}.pt"
            )
            global_checkpoint = {
                "type": ShardedCheckpointType.SHARD_LIST.name,
                "shards_root": self.checkpoint_folder,
                "shards": [
                    self._get_shard_file_path(
                        epoch, shard_id, with_epoch=True, rel_path=True
                    )
                    for shard_id in range(self.world_size)
                ],
            }
            with g_pathmgr.open(checkpoint_path, "wb") as f:
                torch.save(global_checkpoint, f)

    def _should_save_epoch_specific(self, epoch: int) -> bool:
        return self.save_freq > 0 and epoch % self.save_freq == 0

    def _get_shard_file_path(
        self, epoch: int, shard_id: int, with_epoch: bool, rel_path: bool = False
    ) -> str:
        if with_epoch:
            path = os.path.join(
                self.shard_sub_folder,
                f"checkpoint_{epoch}_shard{shard_id}.pt",
            )
        else:
            path = os.path.join(
                self.shard_sub_folder,
                f"checkpoint_shard{shard_id}.pt",
            )
        if not rel_path:
            path = os.path.join(self.checkpoint_folder, path)
        return path


class ShardedCheckpointLoader:
    """
    Helper class to load checkpoints of FSDP models
    """

    def __init__(self):
        self.world_size = torch.distributed.get_world_size()
        _, disk_rank = get_machine_local_and_dist_rank()
        self.worker_id = disk_rank

    def load_resume_checkpoint(self, checkpoint_path: str):
        """
        Allows to resume training for a FSDP model, using sharded checkpoint
        """
        with g_pathmgr.open(checkpoint_path, "rb") as f:
            shard_list_checkpoint = torch.load(f, map_location="cpu")

        checkpoint_type = shard_list_checkpoint.get(
            "type", ShardedCheckpointType.CONSOLIDATED.name
        )
        expected_type = ShardedCheckpointType.SHARD_LIST.name
        if checkpoint_type != expected_type:
            raise ValueError(
                f"Restoring a FSDP model requires a checkpoint of type {expected_type}"
            )

        shard_paths = shard_list_checkpoint["shards"]
        if len(shard_paths) != self.world_size:
            error_msg = f"Restoring a FSDP model requires a checkpoint with identical world size ({self.world_size})"
            raise ValueError(error_msg)

        shard_path = os.path.join(
            os.path.dirname(checkpoint_path), shard_paths[self.worker_id]
        )
        with g_pathmgr.open(shard_path, "rb") as f:
            return torch.load(f, map_location="cpu")

    # Mannat: commenting out since it is unused currently
    # def init_model_from_checkpoint_path(
    #     self, model: FullyShardedDataParallel, checkpoint_path: str, strict: bool = True
    # ):
    #     """
    #     Load an evaluation checkpoint:
    #     - does not load the optimizer state or anything but the model
    #     - support several format of checkpoints (sharded or consolidated)
    #     """
    #     with g_pathmgr.open(checkpoint_path, "rb") as f:
    #         checkpoint = torch.load(f, map_location="cpu")
    #     self.init_model_from_checkpoint(model, checkpoint, strict=strict)

    def init_model_from_checkpoint(
        self,
        model: FullyShardedDataParallel,
        checkpoint: Dict,
        ckpt_state_dict_keys: List[str],
        checkpoint_kernels=None,
        strict: bool = True,
    ):
        checkpoint_type = checkpoint.get(
            "type", ShardedCheckpointType.CONSOLIDATED.name
        )
        if checkpoint_type == ShardedCheckpointType.SHARD_LIST.name:
            # TODO: Add support for kernels
            assert checkpoint_kernels is None
            assert ckpt_state_dict_keys == ["model"]
            self._init_model_from_sharded(model, checkpoint, strict=strict)
        elif checkpoint_type == ShardedCheckpointType.CONSOLIDATED.name:
            self._init_model_from_conso(
                model,
                checkpoint,
                ckpt_state_dict_keys,
                checkpoint_kernels,
                strict=strict,
            )
        else:
            options = [
                ShardedCheckpointType.SHARD_LIST.name,
                ShardedCheckpointType.CONSOLIDATED.name,
            ]
            msg = f"Cannot initialize FSDP model from a {checkpoint_type} checkpoint. Please use one of {options}."
            raise ValueError(msg)

    def _init_model_from_sharded(
        self,
        model: FullyShardedDataParallel,
        checkpoint: Dict[str, Any],
        strict: bool = True,
    ):
        """
        Initialize a FSDP model from a list of of shard checkpoint
        """
        assert checkpoint["type"] == ShardedCheckpointType.SHARD_LIST.name
        relative_shard_path = checkpoint["shards"][self.worker_id]
        root_folder = checkpoint["shards_root"]
        shard_path = os.path.join(root_folder, relative_shard_path)
        with g_pathmgr.open(shard_path, "rb") as f:
            shard_content = torch.load(f, map_location="cpu")
        model.load_local_state_dict(shard_content["model"], strict=strict)

    def _init_model_from_conso(
        self,
        model: FullyShardedDataParallel,
        checkpoint: Dict[str, Any],
        ckpt_state_dict_keys: List[str],
        checkpoint_kernels,
        strict: bool = True,
    ):
        """
        Initialize a FSDP model from a consolidated checkpoint.
        Avoid consolidating the whole FSDP model by calling load_state_dict
        and instead stream the FSDP consolidation as if inside a forward
        """
        state_dict = get_state_dict(checkpoint, ckpt_state_dict_keys)

        if checkpoint_kernels is not None:
            for f in checkpoint_kernels:
                state_dict = f(state_dict=state_dict)

        for path, module in self._recursive_visit(model):
            for param_path, param in module.named_parameters(
                prefix=path, recurse=False
            ):
                self._init_weight_from_state_dict(
                    param_path, param.data, state_dict, strict=strict
                )
            for buffer_path, buffer in module.named_buffers(prefix=path, recurse=False):
                self._init_weight_from_state_dict(
                    buffer_path, buffer.data, state_dict, strict=strict
                )

    @classmethod
    def _init_weight_from_state_dict(
        cls,
        weight_path: str,
        weight: torch.Tensor,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = True,
    ):
        weight_path = cls._clean_path(weight_path)
        checkpoint_weight = state_dict.get(weight_path)
        if checkpoint_weight is None:
            if strict:
                raise ValueError(f"Missing weight: {weight_path}")
            return
        else:
            weight.copy_(checkpoint_weight)

    @classmethod
    def _recursive_visit(cls, model: FullyShardedDataParallel):
        """
        Visit a FSDP model, summoning parameters on the fly
        and releasing them as soon as they are not needed

        This replicates the summoning of parameters as done
        through the forward pass of a FSDP model
        """

        def visit(path, module):
            context = contextlib.nullcontext()
            if isinstance(module, FullyShardedDataParallel):
                context = module.summon_full_params(recurse=False)

            with context:
                yield path, module
                for name, child in module._modules.items():
                    next_path = path + "." + name if path else name
                    yield from visit(next_path, child)

        yield from visit("", model)

    @staticmethod
    def _clean_path(param_path: str):
        fsdp_names = {"_fsdp_wrapped_module", "_fpw_module"}
        return ".".join(
            [split for split in param_path.split(".") if split not in fsdp_names]
        )


def load_state_dict_into_fsdp_model(
    model,
    path_list: List[str],
    ckpt_state_dict_keys: List[str],
    checkpoint_kernels=None,
    strict: bool = True,
):
    """
    Performs checkpoint loading with a variety of pre-processing kernel applied in
    sequence.

    Args:
        checkpoint_path: Path to the checkpoint.
        checkpoint_kernels: A list of checkpoint processing kernels
            to apply in the specified order. Supported kernels include
            `CkptIncludeKernel`, `CkptExcludeKernel`, etc. These kernels are applied
            in the given order.
        ckpt_state_dict_keys: Keys containing the model state dict.
        strict: raise if the state_dict

    Returns: FSDP Model with the matching pre-trained weights loaded.
    """
    assert is_fsdp(model), "FSDPModelWeightInitializer is meant for FSDP models"
    checkpoint = load_checkpoint(path_list)
    loader = ShardedCheckpointLoader()
    loader.init_model_from_checkpoint(
        model, checkpoint, ckpt_state_dict_keys, checkpoint_kernels, strict=strict
    )
    return model


def wrap_with_fsdp(fsdp_settings: FSDPSettings, module: nn.Module):
    """
    Useful wrapper to use inside configuration to wrap modules that have
    no knowledge of FSDP (for instance 'torch.nn' modules like Linear).

    Don't use this for model that use 'wrap_fsdp' inside their definition.
    """
    with enable_fsdp(fsdp_settings):
        return wrap_fsdp_if_enabled(module)


def escape_fsdp(fsdp_settings: FSDPSettings, module: nn.Module):
    """
    Wraps a module with FSDP but disable the re-sharding after forward, which
    effectively makes the layer non-FSDP (it will never be sharded except at init).

    Useful wrapper to use inside configuration.

    Don't use this for model that use 'wrap_fsdp' inside their definition.
    """
    escape_settings = dataclasses.replace(fsdp_settings, reshard_after_forward=False)
    with enable_fsdp(escape_settings):
        return wrap_fsdp_if_enabled(module)


def dont_hook_this_fsdp(x: torch.Tensor):
    """
    FSDP will latch (register backward hooks) on every tensor output of a model
    that has a requires_grad set to True

    If you want to return parameters or anything that should not be
    latched on for the backward, you can either:
    - wrap your output with a dummy wrapper type
    - detach the tensor (option selected here)
    """
    return x.detach()
