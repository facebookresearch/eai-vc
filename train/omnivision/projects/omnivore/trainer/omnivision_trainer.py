# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import contextlib
import copy
import gc
import json
import logging
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from omnivision.optim import construct_optimizer
from omnivore.data.api import Sample

from omnivore.losses import CORE_LOSS_KEY
from omnivore.meters import FINAL_LOGITS_NAME
from omnivore.models.fsdp_model_utils import (
    create_fsdp_grad_scaler,
    setup_fsdp_distributed_training,
    ShardedCheckpointLoader,
    ShardedCheckpointSaver,
)
from omnivore.utils import Phase
from omnivore.utils.model_summary import print_model_summary


def get_supported_dataloader():
    from omnivore.data.airstore_dataset import AirStoreTorchDataLoader
    from omnivore.data.concat_dataset import ConcatDataset
    from omnivore.data.torch_dataset import TorchDataset

    supported_dataloaders = (TorchDataset, ConcatDataset, AirStoreTorchDataLoader)
    try:
        from omnivore.data.webdataset_helpers import WebVisionDatasetBatchedWithLoader

        USE_WEB_DATASET = True
    except ImportError:
        logging.warn(
            "WebVisionDatasetBatchedWithLoader is not supported in this environment."
        )
        USE_WEB_DATASET = False
    if USE_WEB_DATASET:
        supported_dataloaders += (WebVisionDatasetBatchedWithLoader,)

    try:
        from omnivore.data.fb.on_box_dataset import OnBoxDataset

        USE_ONBOX_DATASET = True
    except ImportError:
        logging.warn("OnBoxDataset is not supported in this environment.")
        USE_ONBOX_DATASET = False
    if USE_ONBOX_DATASET:
        supported_dataloaders += (OnBoxDataset,)
    return supported_dataloaders


_SUPPORTED_DATALOADERS = get_supported_dataloader()


from omnivore.losses import wrap_base_loss
from omnivore.train_utils import (
    AverageMeter,
    copy_data_to_device,
    get_amp_type,
    get_machine_local_and_dist_rank,
    get_resume_checkpoint,
    is_dist_avail_and_initialized,
    makedir,
    ProgressMeter,
    set_seeds,
    setup_distributed_backend,
    setup_logging,
)


def chunk_batch_for_accum_steps(batch, accum_steps):
    return [get_chunk_from_data(batch, i, accum_steps) for i in range(accum_steps)]


def get_chunk_from_data(data, chunk_id, num_chunks):
    """
    Recursively splits all the tensors inside the passed data object into num_chunks.
    """
    if isinstance(data, torch.Tensor):
        assert len(data) % num_chunks == 0
        start = (len(data) // num_chunks) * chunk_id
        end = (len(data) // num_chunks) * (chunk_id + 1)
        return data[start:end]
    elif isinstance(data, Mapping):
        return {
            key: get_chunk_from_data(value, chunk_id, num_chunks)
            for key, value in data.items()
        }
    elif isinstance(data, Sequence):
        return [get_chunk_from_data(value, chunk_id, num_chunks) for value in data]
    elif isinstance(data, Sample):
        data_cls = type(data)
        data = data.__dict__
        return data_cls(**get_chunk_from_data(data, chunk_id, num_chunks))
    else:
        return data


@dataclass
class OmnivisionOptimAMPConf:
    enabled: bool = False
    amp_dtype: str = "float16"


@dataclass
class OmnivisionOptimConf:
    optimizer: torch.optim.Optimizer = None
    options: Optional[Dict[str, Any]] = None
    param_group_modifiers: Optional[List] = None
    amp: Optional[Dict[str, Any]] = None
    gradient_clip: Any = None
    gradient_logger: Any = None

    def __post_init__(self):
        # amp
        if not isinstance(self.amp, OmnivisionOptimAMPConf):
            if self.amp is None:
                self.amp = {}
            assert isinstance(self.amp, Mapping)
            self.amp = OmnivisionOptimAMPConf(**self.amp)


@dataclass
class OmnivisionDistributedConf:
    backend: Optional[str] = None  # inferred from accelerator type
    comms_dtype: Optional[str] = None
    find_unused_parameters: bool = False
    timeout_mins: int = 30


@dataclass
class OmnivisionCudaConf:
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = True
    allow_tf32: bool = False


@dataclass
class OmnivisionCheckpointConf:
    save_dir: str
    save_freq: int
    model_weight_initializer: Any = None


@dataclass
class OmnivisionEMAConf:
    enabled: bool = False
    # The EMA model will be "warmed-up" to exactly match the original model for warmup
    # period during the training, effectively having a decay of 1.0 when
    # self.where < warmup.
    warmup: float = None
    decay: float = None
    # EMA will be updated every freq steps
    freq: int = 1


@dataclass
class OmnivisionLoggingConf:
    log_dir: str
    log_freq: int  # In iterations
    tensorboard_writer: Any
    # Separate writer for saving feature embeddings for tSNE
    # visualization. Not storing in the same files as above
    # since these might get too large and this would allow for
    # deleting these tensorboard files separately.
    tensorboard_embedding_writer: Optional[Any] = None


class OmnivisionTrainer(object):
    """
    Omnivision Trainer supporting the DDP and FSDP training strategies.
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,  # the order of these args can change at any time, so they are keyword-only
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        accelerator: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        optim: Optional[Dict[str, Any]] = None,
        meters: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        ema: Optional[Dict[str, Any]] = None,
        strategy: str = "ddp",
        fsdp_settings: Optional[Dict[str, Any]] = None,
    ):
        ## TODO: Re-factor to expose train_step as target.
        ## TODO: Support for Sync batchnorm.

        self._setup_env_variables(env_variables)
        self._print_paths_to_code()
        self._print_env()

        self.data_conf = data
        self.model_conf = model
        self.logging_conf = OmnivisionLoggingConf(**logging)
        self.checkpoint_conf = OmnivisionCheckpointConf(**checkpoint)
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.optim_conf = OmnivisionOptimConf(**optim or {})
        self.meters_conf = meters
        self.loss_conf = loss
        distributed = OmnivisionDistributedConf(**distributed or {})
        cuda = OmnivisionCudaConf(**cuda or {})
        self.ema_conf = OmnivisionEMAConf(**ema or {})
        self.where = 0.0

        self._infer_distributed_backend_if_none(distributed, accelerator)
        self.strategy = strategy
        self.fsdp_settings = fsdp_settings
        self._check_strategy_consistency()

        self._setup_device(accelerator)

        makedir(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.local_rank,
        )
        # TODO: Enable separate seed setting for each data worker.
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)

        self._setup_torch_dist_and_backend(cuda, distributed)

        assert (
            is_dist_avail_and_initialized()
        ), "Torch distributed needs to be initialized before calling the trainer."

        self._setup_components()  # Except Optimizer everything is setup here.
        self._move_to_device()
        self._construct_optimizer()
        self._setup_ema_if_enabled()
        self._setup_dataloaders()

        if self._is_fsdp_training():
            self._setup_distributed_training(distributed, accelerator)
            self.load_checkpoint()
        else:
            self.load_checkpoint()
            self._setup_distributed_training(distributed, accelerator)
        dist.barrier()

    def _print_paths_to_code(self):
        import omnivision
        import omnivore

        print(f"Path to omnivision: {omnivision.__file__}")
        print(f"Path to omnivore: {omnivore.__file__}")

    def _print_env(self):
        print(f"Environment:\n{json.dumps(dict(os.environ), sort_keys=True, indent=2)}")

    def _get_meters(self, phase_filters=None):
        if self.meters is None:
            return {}
        meters = {}
        for phase, phase_meters in self.meters.items():
            if phase_filters is not None and phase not in phase_filters:
                continue
            for key, key_meters in phase_meters.items():
                for name, meter in key_meters.items():
                    meters[f"{phase}_{key}/{name}"] = meter
        return meters

    def _infer_distributed_backend_if_none(self, distributed_conf, accelerator):
        if distributed_conf.backend is None:
            distributed_conf.backend = "nccl" if accelerator == "cuda" else "gloo"

    def _setup_env_variables(self, env_variables_conf) -> None:
        if env_variables_conf is not None:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value

    def _setup_torch_dist_and_backend(self, cuda_conf, distributed_conf) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32

        setup_distributed_backend(
            distributed_conf.backend, distributed_conf.timeout_mins
        )

    def _setup_device(self, accelerator):
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if accelerator == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif accelerator == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported accelerator: {accelerator}")

    def _is_fsdp_training(self) -> bool:
        return self.strategy == "fsdp"

    def _check_strategy_consistency(self):
        error_msg = "FSDP settings should be set if and only if FSDP training strategy is selected"
        has_fsdp_settings = self.fsdp_settings is not None
        assert self._is_fsdp_training() == has_fsdp_settings, error_msg

    def _setup_distributed_training(self, distributed_conf, accelerator):
        # We need a specific wrapping for FSDP training (and not use DDP)
        if self._is_fsdp_training():
            self.model = setup_fsdp_distributed_training(self.fsdp_settings, self.model)
            return

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if accelerator == "cuda" else [],
            find_unused_parameters=distributed_conf.find_unused_parameters,
        )

        if distributed_conf.comms_dtype is not None:  # noqa

            from torch.distributed.algorithms import ddp_comm_hooks

            amp_type = get_amp_type(distributed_conf.comms_dtype)
            if amp_type == torch.bfloat16:
                hook = ddp_comm_hooks.default_hooks.bf16_compress_hook
                logging.info("Enabling bfloat16 grad communication")
            else:
                hook = ddp_comm_hooks.default_hooks.fp16_compress_hook
                logging.info("Enabling fp16 grad communication")
            process_group = None
            self.model.register_comm_hook(process_group, hook)

    def _setup_ema_if_enabled(self):
        if not self.ema_conf.enabled:
            return
        assert 0 <= self.ema_conf.decay <= 1
        assert (
            0 < self.ema_conf.warmup <= 1
        ), "Warm up has to be > 0 to ensure correct init"
        assert not isinstance(self.model, nn.parallel.DistributedDataParallel)
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False
        if Phase.VAL in self.meters:
            self.meters[Phase.VAL_EMA] = copy.deepcopy(self.meters[Phase.VAL])

    def _move_to_device(self):
        logging.info(
            f"Moving components to device {self.device} and local rank {self.local_rank}."
        )
        self.model.to(self.device)

        if self.loss:
            copy_data_to_device(self.loss, self.device)
        if self.scaler:
            copy_data_to_device(self.scaler, self.device)
        for meter in self._get_meters().values():
            meter.set_sync_device(self.device)

        logging.info(
            f"Done moving components to device {self.device} and local rank {self.local_rank}."
        )

    def save_checkpoint(self, epoch):
        checkpoint_folder = self.checkpoint_conf.save_dir

        # FSDP checkpoints are saved on all ranks as each worker has a different
        # shard of the parameters, optimizer state...
        if self._is_fsdp_training():
            makedir(checkpoint_folder)
            saver = ShardedCheckpointSaver(
                checkpoint_folder,
                save_freq=self.checkpoint_conf.save_freq,
            )
            saver.save_checkpoint(
                model=self.model,
                optimizer=self.optim.optimizer,
                loss=self.loss,
                epoch=epoch,
                steps=self.steps,
                scaler=self.scaler,
                train_dataset_state=self._get_train_dataset_checkpoint_state(),
            )
            return

        # DDP checkpoints are only saved on rank 0 (all workers are identical)
        if self.distributed_rank != 0:
            return

        makedir(checkpoint_folder)
        checkpoint_paths = [os.path.join(checkpoint_folder, "checkpoint.pt")]
        if (
            self.checkpoint_conf.save_freq > 0
            and int(epoch) % self.checkpoint_conf.save_freq == 0
        ):
            checkpoint_paths.append(
                os.path.join(checkpoint_folder, f"checkpoint_{int(epoch)}.pt")
            )

        checkpoint = {
            "model": self.model.module.state_dict(),
            "optimizer": self.optim.optimizer.state_dict(),
            "epoch": epoch,
            "loss": self.loss.state_dict(),
            "steps": self.steps,
        }
        if self.optim_conf.amp.enabled:
            checkpoint["scaler"] = self.scaler.state_dict()

        if self.ema_conf.enabled:
            checkpoint["ema_model"] = self.ema_model.state_dict()

        train_dataset_state = self._get_train_dataset_checkpoint_state()
        if train_dataset_state is not None:
            checkpoint["train_dataset"] = train_dataset_state

        for checkpoint_path in checkpoint_paths:
            with g_pathmgr.open(checkpoint_path, "wb") as f:
                torch.save(checkpoint, f)

    def _get_train_dataset_checkpoint_state(self):
        if self.train_dataset is not None:
            return self.train_dataset.get_checkpoint_state()
        return None

    def load_checkpoint(self):
        ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)

        if ckpt_path is None:
            # Loading pre-trained model weights
            model_weight_initializer = instantiate(
                self.checkpoint_conf.model_weight_initializer
            )
            if model_weight_initializer is not None:
                logging.info(
                    f"Loading pretrained checkpoint from {self.checkpoint_conf.model_weight_initializer}"
                )
                self.model = model_weight_initializer(model=self.model)
        else:
            # Resuming from previous training checkpoint
            logging.info(f"Resuming training from {ckpt_path}")
            if self._is_fsdp_training():
                loader = ShardedCheckpointLoader()
                checkpoint = loader.load_resume_checkpoint(ckpt_path)
                self.model.load_local_state_dict(checkpoint["model"], strict=True)

            else:
                with g_pathmgr.open(ckpt_path, "rb") as f:
                    checkpoint = torch.load(f, map_location="cpu")
                self.model.load_state_dict(checkpoint["model"], strict=True)

            self.optim.optimizer.load_state_dict(checkpoint["optimizer"])
            self.loss.load_state_dict(checkpoint["loss"], strict=True)
            self.epoch = checkpoint["epoch"]
            self.steps = checkpoint["steps"]

            if self.optim_conf.amp.enabled and "scaler" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler"])

            if self.ema_conf.enabled:
                self.ema_model.load_state_dict(checkpoint["ema_model"])

            if "train_dataset" in checkpoint and self.train_dataset is not None:
                self.train_dataset.load_checkpoint_state(checkpoint["train_dataset"])

    def run(self):
        assert self.mode in ["train", "train_only", "val"]
        if self.mode == "train":
            self.run_train()
            self.run_val()
        elif self.mode == "val":
            self.run_val()
        elif self.mode == "train_only":
            self.run_train()

    def _setup_dataloaders(self):
        self.train_dataset = None
        self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(self.data_conf.get(Phase.VAL, None))
            if self.val_dataset:
                assert isinstance(
                    self.val_dataset, _SUPPORTED_DATALOADERS
                ), f"Unsuported Val dataloader: {type(self.val_dataset).__name__}"

        if self.mode in ["train", "train_only"]:
            self.train_dataset = instantiate(self.data_conf.train)
            assert isinstance(
                self.train_dataset, _SUPPORTED_DATALOADERS
            ), f"Unsuported Train dataloader: {type(self.train_dataset).__name__}"

    def run_train(self):
        # loop
        while self.epoch < self.max_epochs:

            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch))
            outs = self.train_epoch(dataloader)
            del dataloader
            gc.collect()
            self.logger.log_dict(outs, self.epoch)  # Logged only on rank 0

            # log train to text file.
            if self.distributed_rank == 0:
                with g_pathmgr.open(
                    os.path.join(self.logging_conf.log_dir, "train_stats.json"),
                    "a",
                ) as f:
                    f.write(json.dumps(outs) + "\n")

            # Save checkpoint before validating
            self.save_checkpoint(self.epoch + 1)

            # Run val, not running on last epoch since will run after the
            # loop anyway
            if (
                self.epoch % self.val_epoch_freq == 0
                and self.epoch < self.max_epochs - 1
            ):
                self.run_val()

            self.epoch += 1
        # epoch was incremented in the loop but the val step runs out of the loop
        self.epoch -= 1

    def run_val(self):
        if not self.val_dataset:
            return

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch))
        outs = self.val_epoch(dataloader)
        del dataloader
        gc.collect()
        self.logger.log_dict(outs, self.epoch)  # Logged only on rank 0

        if self.distributed_rank == 0:
            with g_pathmgr.open(
                os.path.join(self.logging_conf.log_dir, "val_stats.json"),
                "a",
            ) as f:
                f.write(json.dumps(outs) + "\n")

    def val_epoch(self, val_loader):
        batch_time = AverageMeter("Batch Time", self.device, ":.2f")
        data_time = AverageMeter("Data Time", self.device, ":.2f")
        mem = AverageMeter("Mem (GB)", self.device, ":.2f")

        iters_per_epoch = len(val_loader)

        curr_phases = [Phase.VAL]
        if self.ema_conf.enabled:
            curr_phases.append(Phase.VAL_EMA)

        progress = ProgressMeter(
            iters_per_epoch,
            [batch_time, data_time, mem],
            self._get_meters(curr_phases),
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        self.model.eval()
        if hasattr(self.model.module, "on_validation_epoch_start"):
            self.model.module.on_validation_epoch_start()

        end = time.time()

        limit_val_batches = (
            iters_per_epoch
            if self.limit_val_batches is None
            else self.limit_val_batches
        )

        for data_iter, batch in enumerate(val_loader):

            if data_iter > limit_val_batches:
                break

            # measure data loading time
            data_time.update(time.time() - end)

            key, batch = self._process_batch(batch, Phase.VAL)
            batch = copy_data_to_device(batch, self.device)

            # compute output
            with torch.no_grad():
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=get_amp_type(self.optim_conf.amp.amp_dtype),
                ):
                    for p in curr_phases:
                        self._step(
                            batch,
                            key,
                            phase=p,
                        )

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

            # Log progress meters.
            for progress_meter in progress.meters:
                self.logger.log(
                    os.path.join("Step_Stats", Phase.VAL, progress_meter.name),
                    progress_meter.val,
                    self.steps[Phase.VAL],
                )

        if hasattr(self.model.module, "on_validation_epoch_end"):
            self.model.module.on_validation_epoch_end()

        logging.info("Synchronizing meters")
        out_dict = {}
        for key, meter in self._get_meters(curr_phases).items():
            meter_output = meter.compute_synced()
            for meter_subkey, meter_value in meter_output.items():
                out_dict[os.path.join("Meters_val", key, meter_subkey)] = meter_value
        for phase in curr_phases:
            out_dict.update(self._get_trainer_state(phase))
        self._reset_meters(curr_phases)
        logging.info(f"Meters: {out_dict}")
        return out_dict

    def _get_trainer_state(self, phase):
        return {
            "Trainer/where": self.where,
            "Trainer/epoch": self.epoch,
            f"Trainer/steps_{phase}": self.steps[phase],
        }

    def train_epoch(self, train_loader):
        batch_time = AverageMeter("Batch Time", self.device, ":.2f")
        data_time = AverageMeter("Data Time", self.device, ":.2f")
        mem = AverageMeter("Mem (GB)", self.device, ":.2f")
        phase = Phase.TRAIN

        iters_per_epoch = len(train_loader)

        loss_names = []
        for key in self.loss.keys():
            loss_names.append(f"Losses/{phase}_{key}_loss")

        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names]
        )

        # TODO: Track optimizer params (LR, WD,) etc.
        progress = ProgressMeter(
            iters_per_epoch,
            [batch_time, data_time, mem, *loss_mts.values()],
            self._get_meters([phase]),
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()

        if hasattr(self.model.module, "on_train_epoch_start"):
            self.model.module.on_train_epoch_start()

        end = time.time()

        limit_train_batches = (
            iters_per_epoch
            if self.limit_train_batches is None
            else self.limit_train_batches
        )

        for data_iter, batch in enumerate(train_loader):

            if data_iter > limit_train_batches:
                break

            # measure data loading time
            data_time.update(time.time() - end)

            key, batch = self._process_batch(batch, phase)
            batch = copy_data_to_device(batch, self.device)

            accum_steps = batch.accum_steps
            chunked_batches = chunk_batch_for_accum_steps(batch, accum_steps)

            self.optim.zero_grad()

            for i, chunked_batch in enumerate(chunked_batches):
                ddp_context = (
                    self.model.no_sync()
                    if i < accum_steps - 1
                    else contextlib.nullcontext()
                )

                with ddp_context:
                    with torch.cuda.amp.autocast(
                        enabled=self.optim_conf.amp.enabled,
                        dtype=get_amp_type(self.optim_conf.amp.amp_dtype),
                    ):
                        loss_dict, batch_size = self._step(
                            chunked_batch,
                            key,
                            phase=phase,
                        )

                    assert len(loss_dict) == 1
                    loss_key, loss = loss_dict.popitem()

                    if not math.isfinite(loss.item()):
                        error_msg = f"Loss is {loss.item()}, stopping training"
                        logging.error(error_msg)
                        raise ValueError(error_msg)

                    loss /= accum_steps
                    self.scaler.scale(loss).backward()

                    loss_mts[loss_key].update(loss.item(), batch_size)

            # compute gradient and do SGD step
            exact_epoch = self.epoch + float(data_iter) / iters_per_epoch
            self.where = float(exact_epoch) / self.max_epochs
            assert self.where <= 1 + self.EPSILON
            # TODO: Fixme, create on-box dataloader wrapper to handle this.
            # Happens with Auto-onbox dataloaders where length retuned is always len(actual_dataloader)-1
            if self.where < 1.0:
                self.optim.step_schedulers(self.where)
            else:
                logging.warn(
                    f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                )

            # Log schedulers
            for i, param_group in enumerate(self.optim.optimizer.param_groups):
                for option in self.optim.schedulers[i]:
                    self.logger.log(
                        os.path.join("Optim", str(i), option),
                        param_group[option],
                        self.steps[phase],
                    )

            # Clipping gradients
            if self.gradient_clipper is not None:
                self.scaler.unscale_(self.optim.optimizer)
                self.gradient_clipper(model=self.model)

            # Looking at the scale of the gradients after clipping
            if self.gradient_logger is not None:
                self.gradient_logger(
                    self.model, rank=self.distributed_rank, where=self.where
                )

            # Optimizer step: the scaler will make sure gradients are not
            # applied if the gradients are infinite
            self.scaler.step(self.optim.optimizer)
            self.scaler.update()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

            # Log progress meters.
            for progress_meter in progress.meters:
                self.logger.log(
                    os.path.join("Step_Stats", phase, progress_meter.name),
                    progress_meter.val,
                    self.steps[phase],
                )

        logging.info("Synchronizing meters")
        out_dict = {}
        for key, meter in self._get_meters([phase]).items():
            meter_output = meter.compute_synced()
            for meter_subkey, meter_value in meter_output.items():
                out_dict[os.path.join("Meters_train", key, meter_subkey)] = meter_value
        for k, v in loss_mts.items():
            out_dict[k] = v.avg
        out_dict.update(self._get_trainer_state(phase))
        logging.info(f"Losses and meters: {out_dict}")
        self._reset_meters([phase])
        return out_dict

    def _compute_meters(
        self,
        # Could be the logits tensor, or the feature map, with a mapping from
        # feature name to tensor (eg in MultiModalZeroShotWithTextTargetsWrapper)
        pred: Union[torch.Tensor, Mapping[str, torch.Tensor]],
        label: torch.Tensor,
        phase: str,
        key: str,
    ) -> Dict[str, torch.Tensor]:
        if phase not in self.meters:
            return {}
        if key not in self.meters[phase]:
            return {}
        meters_dict = self.meters[phase][key]
        for meter_key, meter in meters_dict.items():
            meter.update(pred, label)
            meter_output = meter.compute()
            for meter_subkey, meter_subval in meter_output.items():
                self.logger.log(
                    os.path.join("Step_Meters", phase, key, meter_key, meter_subkey),
                    meter_subval,
                    self.steps[phase],
                )

    def _reset_meters(self, phases: str) -> None:
        for meter in self._get_meters(phases).values():
            meter.reset()

    def _setup_components(self):
        logging.info("Setting up components: Model, loss, optim, meters etc.")
        self.epoch = 0
        self.steps = {Phase.TRAIN: 0, Phase.VAL: 0}
        if self.ema_conf.enabled:
            self.steps[Phase.VAL_EMA] = 0

        self.logger = instantiate(self.logging_conf.tensorboard_writer)
        self.embedding_logger = None
        if self.logging_conf.tensorboard_embedding_writer:
            self.embedding_logger = instantiate(
                self.logging_conf.tensorboard_embedding_writer
            )

        self.model = instantiate(self.model_conf, _convert_="all")
        print_model_summary(self.model)

        self.loss = {
            key: wrap_base_loss(el)
            for (key, el) in instantiate(self.loss_conf, _convert_="all").items()
        }
        self.loss = nn.ModuleDict(self.loss)

        self.meters = {}
        if self.meters_conf:
            self.meters = instantiate(self.meters_conf, _convert_="all")

        # FDSP needs a different Gradient Scaler than DDP
        if self._is_fsdp_training():
            self.scaler = create_fsdp_grad_scaler(enabled=self.optim_conf.amp.enabled)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.optim_conf.amp.enabled)

        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)
        self.gradient_logger = instantiate(self.optim_conf.gradient_logger)

        logging.info("Finished setting up components: Model, loss, optim, meters etc.")

    def _construct_optimizer(self):
        self.optim = (
            None
            if self.optim_conf is None
            else construct_optimizer(
                self.model,
                self.optim_conf.optimizer,
                self.optim_conf.options,
                self.optim_conf.param_group_modifiers,
            )
        )

    def _process_batch(self, batch, phase):
        assert isinstance(batch, Mapping)
        assert all(isinstance(v, Sample) for v in batch.values())
        assert len(batch) == 1
        key, batch_sample = batch.popitem()
        self.logger.log(
            f"Data/BatchValid/{phase}/{key}",
            batch_sample.data_valid.float().mean().item(),
            self.steps[phase],
        )
        return key, batch_sample

    def _compute_ema_if_enabled(self):
        if not self.ema_conf.enabled or (
            self.steps[Phase.TRAIN] % self.ema_conf.freq > 0
        ):
            return
        decay = 1.0 if self.where <= self.ema_conf.warmup else self.ema_conf.decay
        model_state = self.model.module.state_dict()
        ema_model_state = self.ema_model.state_dict()
        with torch.no_grad():
            for name in model_state:
                if not isinstance(model_state[name], torch.Tensor):
                    raise TypeError(
                        f"Unexpected value in model state for key {name}:"
                        f" {model_state[name]}"
                    )
                if not ema_model_state[name].is_floating_point():
                    if self.steps[Phase.TRAIN] == 0:
                        logging.warning(
                            f"EMA will be skipping key: {name} since it"
                            f" is of type: {ema_model_state[name].dtype}"
                        )
                    continue
                ema_model_state[name].copy_(
                    (1.0 - decay) * ema_model_state[name] + decay * model_state[name]
                )

    def _add_embeddings_tb(self, y_hat, dataset_key, phase):
        if self.embedding_logger is None:
            return
        if not isinstance(y_hat, Mapping):
            # This is a val sample likely, where we convert
            # the input to logits for 0-shot eval..
            assert isinstance(y_hat, torch.Tensor)
            y_hat = {FINAL_LOGITS_NAME: y_hat}
        for embedding_key in y_hat.keys():
            # embedding_key is like "vision_embed" etc
            tensor = y_hat[embedding_key]
            self.embedding_logger.add_embedding(
                tensor,
                dataset=dataset_key,
                embed_name=embedding_key,
                phase=phase,
                epoch=self.epoch,
            )

    def _log_loss_detailed(self, loss, loss_str, step):
        core_loss = loss.pop(CORE_LOSS_KEY)
        for k in loss:
            log_str = os.path.join(loss_str, k)
            self.logger.log(log_str, loss[k], step)
        return core_loss

    def _step(self, batch: Any, key: str, phase: str):
        model = self.ema_model if phase == Phase.VAL_EMA else self.model
        y_hat = model({key: batch}, **batch.model_fwd_kwargs)
        assert isinstance(y_hat, Mapping)
        assert len(y_hat) == 1, str(len(y_hat))
        key, y_hat = y_hat.popitem()
        loss = None
        batch_size = batch.label.shape[0]
        loss_str = f"Losses/{phase}_{key}_loss"
        if phase == Phase.TRAIN:
            loss, y_hat = self.loss[key](y_hat, batch)
            loss_log_str = os.path.join("Step_Losses", loss_str)

            if isinstance(loss, dict):
                # loss contains multiple sub-components we wish to log
                loss = self._log_loss_detailed(loss, loss_log_str, self.steps[phase])

            self.logger.log(
                loss_log_str,
                loss,
                self.steps[phase],
            )
            self._compute_ema_if_enabled()

        self._compute_meters(y_hat, batch.label, phase, key)
        self._add_embeddings_tb(y_hat, key, phase)

        self.steps[phase] += 1

        return {loss_str: loss}, batch_size
