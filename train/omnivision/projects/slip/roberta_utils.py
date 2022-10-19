# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Modified from - https://github.com/pytorch/text/blob/00267737e2214e1b01ebc8b2cf68a2d9038407a0/torchtext/models/roberta/bundler.py#L25
# Includes all Roberta definitions.

import logging
import re
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch.nn import Module
from torchtext.models.roberta.model import RobertaEncoderConf, RobertaModel

logger = logging.getLogger(__name__)


@dataclass
class RobertaBaseEncoderConf(RobertaEncoderConf):
    vocab_size: int = 50265
    embedding_dim: int = 768
    ffn_dimension: int = 3072
    padding_idx: int = 1
    max_seq_len: int = 514
    num_attention_heads: int = 12
    num_encoder_layers: int = 12
    dropout: float = 0.1
    scaling: Optional[float] = None
    normalize_before: bool = False


@dataclass
class RobertaLargeEncoderConf(RobertaEncoderConf):
    embedding_dim: int = 1024
    ffn_dimension: int = 4096
    num_attention_heads: int = 16
    num_encoder_layers: int = 24


def _is_head_available_in_checkpoint(checkpoint, head_state_dict):
    # ensure all keys are present
    return all(key in checkpoint.keys() for key in head_state_dict.keys())


def build_roberta_model(
    head: Optional[Module] = None,
    checkpoint_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    override_checkpoint_head: bool = False,
    encoder_conf: RobertaEncoderConf = None,
    strict: bool = False,
    freeze_encoder: bool = False,
) -> RobertaModel:
    """Class builder method
    Args:
        encoder_conf (RobertaEncoderConf): An instance of class RobertaEncoderConf that defined the encoder configuration
        head (nn.Module): A module to be attached to the encoder to perform specific task. (Default: ``None``)
        checkpoint (str or Dict[str, torch.Tensor]): Path to or actual model state_dict. state_dict can have partial weights i.e only for encoder. (Default: ``None``)
        override_checkpoint_head (bool): Override the checkpoint's head state dict (if present) with provided head state dict. (Default: ``False``)
        strict (bool): Passed to :func: `torch.nn.Module.load_state_dict` method. (Default: ``True``)
    """
    if encoder_conf is None:
        encoder_conf = RobertaEncoderConf()  # Roberta base
    model = RobertaModel(encoder_conf, head, freeze_encoder=False)

    if checkpoint_state_dict is not None:
        if head is not None:
            regex = re.compile(r"^head\.")
            head_state_dict = {
                k: v for k, v in model.state_dict().items() if regex.findall(k)
            }
            # If checkpoint does not contains head_state_dict, then we augment the checkpoint with user-provided head state_dict
            if (
                not _is_head_available_in_checkpoint(
                    checkpoint_state_dict, head_state_dict
                )
                or override_checkpoint_head
            ):
                checkpoint_state_dict.update(head_state_dict)

        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint_state_dict, strict=strict
        )
        err = "State key mismatch."
        if unexpected_keys:
            err += f" Unexpected keys: {unexpected_keys}."
        if missing_keys:
            err += f" Missing keys: {missing_keys}."
        if unexpected_keys or missing_keys:
            if not unexpected_keys and not strict:
                logging.warning(err)
            else:
                raise KeyError(err)
        print("Error Message:", err)

    return model
