# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# Code borrowed from TLC - https://www.internalfb.com/code/fbsource/fbcode/pytorch/tlc/torchtlc/loggers/tensorboard.py
import atexit
import gc
import logging
import os
import uuid
from typing import Any, Dict, Optional, Union

import torch

from numpy import ndarray
from omnivore.train_utils import get_machine_local_and_dist_rank, makedir
from omnivore.utils import Phase
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

Scalar = Union[Tensor, ndarray, int, float]


def _setup_wandb(log_dir: str):
    try:
        logging.info(f"initializing wandb...")
        import wandb

        resume = "allow"
        wandb_id = wandb.util.generate_id()

        wandb_filename = os.path.join(log_dir, "wandb", "wandb_id.txt")
        if os.path.exists(wandb_filename):
            # if file exists, then we are resuming from a previous eval
            resume = "must"
            with open(wandb_filename, "r") as file:
                wandb_id = file.read().rstrip("\n")
        else:
            # save wandb file
            os.makedirs(os.path.dirname(wandb_filename), exist_ok=True)
            with open(wandb_filename, "w") as file:
                file.write(wandb_id)
        wandb.init(
            project="omnivision",
            entity="eai-foundations",
            sync_tensorboard=True,
            resume=resume,
            id=wandb_id,
        )
        logging.info(f"initializing wandb... done!")
    except Exception as e:
        logging.warning(f"could not initialize wandb: {e}")


def make_tensorboard_logger(log_dir: str, wandb=False, **writer_kwargs: Any):

    makedir(log_dir)

    if log_dir.startswith("manifold://"):
        from fblearner.flow.util.visualization_utils import (
            log_creation_event,
            summary_writer,
        )

        # TODO: Add support for vis_metrics_from_writer for flow GUI
        summary_writer_method = summary_writer
        tensorboard_url = f"https://internalfb.com/intern/tensorboard/?dir={log_dir}"
        logging.info(f"View TensorBoard logs at: {tensorboard_url}")

        _, rank = get_machine_local_and_dist_rank()
        if rank == 0:
            log_creation_event(log_dir)

    else:
        summary_writer_method = SummaryWriter

    if wandb and get_machine_local_and_dist_rank()[1] == 0:
        _setup_wandb(log_dir)

    return TensorBoardLogger(
        path=log_dir, summary_writer_method=summary_writer_method, **writer_kwargs
    )


# TODO: Expose writer building in configs.
# TODO: How often to flush? flush_secs? max_queue?
class TensorBoardWriterWrapper(object):
    """
    A wrapper around a SummaryWriter object.
    """

    def __init__(
        self,
        path: str,
        *args: Any,
        filename_suffix: str = None,
        summary_writer_method: Any = SummaryWriter,
        **kwargs: Any,
    ) -> None:
        """Create a new TensorBoard logger.
        On construction, the logger creates a new events file that logs
        will be written to.  If the environment variable `RANK` is defined,
        logger will only log if RANK = 0.

        NOTE: If using the logger with distributed training:
        - This logger can call collective operations
        - Logs will be written on rank 0 only
        - Logger must be constructed synchronously *after* initializing distributed process group.

        Args:
            path (str): path to write logs to
            *args, **kwargs: Extra arguments to pass to SummaryWriter
        """
        self._writer: Optional[SummaryWriter] = None
        _, self._rank = get_machine_local_and_dist_rank()
        self._path: str = path
        if self._rank == 0:
            logging.info(
                f"TensorBoard SummaryWriter instantiated. Files will be stored in: {path}"
            )
            self._writer = summary_writer_method(
                log_dir=path,
                *args,
                filename_suffix=filename_suffix or str(uuid.uuid4()),
                **kwargs,
            )
        else:
            logging.debug(
                f"Not logging meters on this host because env RANK: {self._rank} != 0"
            )
        atexit.register(self.close)

    @property
    def writer(self) -> Optional[SummaryWriter]:
        return self._writer

    @property
    def path(self) -> str:
        return self._path

    def flush(self) -> None:
        """Writes pending logs to disk."""

        if not self._writer:
            return

        self._writer.flush()

    def close(self) -> None:
        """Close writer, flushing pending logs to disk.
        Logs cannot be written after `close` is called.
        """

        if not self._writer:
            return

        self._writer.close()
        self._writer = None


class TensorBoardLogger(TensorBoardWriterWrapper):
    """
    A simple logger for TensorBoard.
    """

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        """Add multiple scalar values to TensorBoard.

        Args:
            payload (dict): dictionary of tag name and scalar value
            step (int, Optional): step value to record
        """
        if not self._writer:
            return
        for k, v in payload.items():
            self.log(k, v, step)

    def log(self, name: str, data: Scalar, step: int) -> None:
        """Add scalar data to TensorBoard.

        Args:
            name (string): tag name used to group scalars
            data (float/int/Tensor): scalar data to log
            step (int, optional): step value to record
        """
        if not self._writer:
            return
        self._writer.add_scalar(name, data, global_step=step, new_style=True)

    def log_hparams(
        self, hparams: Dict[str, Scalar], meters: Dict[str, Scalar]
    ) -> None:
        """Add hyperparameter data to TensorBoard.

        Args:
            hparams (dict): dictionary of hyperparameter names and corresponding values
            meters (dict): dictionary of name of meter and corersponding values
        """
        if not self._writer:
            return
        self._writer.add_hparams(hparams, meters)


class TensorBoardEmbeddingLogger(TensorBoardWriterWrapper):
    """
    A logger for TB to store embeddings.
    TODO: Figure a way to overwrite the features, so it's not storing a ton
    of embeddings
    """

    def __init__(
        self,
        *args,
        # Will only store these phases
        phases_to_store=(Phase.TRAIN, Phase.VAL),
        # Will ignore these embeddings
        feature_names_to_skip=("logits",),
        # Will randomly sub-select these many points from
        # GPU 0's data points due to memory/viz limits
        max_points=1e4,
        **kwargs,
    ):
        super().__init__(
            *args, filename_suffix=f"{str(uuid.uuid4())}_embeddings", **kwargs
        )
        self.phases_to_store = phases_to_store
        self.feature_names_to_skip = feature_names_to_skip
        self.max_points = int(max_points)
        # These will get set as the data comes in. At any given
        # time, this logger will be used with only 1
        # (epoch, phase).
        self.cur_epoch = None
        self.cur_phase = None
        # This will store embeddings for a given
        # {"dataset/embed_name": Tensor}
        self.embeddings: Dict[str, Tensor] = {}

    def add_embedding(
        self,
        tensor: Tensor,
        dataset: str,
        embed_name: str,
        phase: str,
        epoch: int,
    ):
        """
        Args:
            tensor: (B, C) tensor of embedding features
            dataset: The name of the dataset this feature comes from
            embed_name: The embedding name (eg vision_embed etc)
            phase: train, val etc
            global_step: the
        """
        if phase not in self.phases_to_store:
            return
        if embed_name in self.feature_names_to_skip:
            return
        if self.cur_phase is not None and (
            phase != self.cur_phase or epoch != self.cur_epoch
        ):
            assert (
                self.cur_epoch is not None
            ), "Both cur_phase and cur_epoch are either None or not None"
            self.dump_embeddings_and_clear()
        self.cur_epoch = epoch
        self.cur_phase = phase
        self.append_embeddings(f"{dataset}/{embed_name}", tensor)

    def dump_embeddings_and_clear(self):
        """Write out whatever embeddings collected so far to tb writer.
        Since each write to the tensorboard is taken as a separate entry, combine them before.
        """
        if not self._writer:
            return
        all_keys = []
        all_vals = []
        for key, val in self.embeddings.items():
            all_keys += [key] * val.size(0)
            all_vals.append(val)
            logging.warning(
                "Embedding writer: Dumping %d features from %s (phase/ep %s/%d)",
                val.size(0),
                key,
                self.cur_phase,
                self.cur_epoch,
            )
        self._writer.add_embedding(
            torch.vstack(all_vals),
            metadata=all_keys,
            tag=f"{self.cur_phase}/{self.cur_phase}",
        )
        del self.embeddings
        gc.collect()  # Just to be doubly sure
        self.embeddings = {}

    def append_embeddings(self, key, features):
        features = features.detach().cpu()
        assert features.numel() > 0
        if key not in self.embeddings:
            self.embeddings[key] = features
        else:
            self.embeddings[key] = torch.vstack([self.embeddings[key], features])
        # Now to ensure we don't overshoot the max_points,
        # we randomly subsample the features. Not ideal as
        # this is biasing the features to the ones that get
        # added later on, but no easy way to handle that...
        if self.embeddings[key].size(0) > self.max_points:
            perm = torch.randperm(self.embeddings[key].size(0))
            perm, _ = torch.sort(perm[: self.max_points])
            self.embeddings[key] = self.embeddings[key][perm, :]

    def close(self) -> None:
        logging.warning(
            "Closing embedding writer, storing features in %s/%s so far",
            self.cur_phase,
            self.cur_epoch,
        )
        self.dump_embeddings_and_clear()
        super().close()
