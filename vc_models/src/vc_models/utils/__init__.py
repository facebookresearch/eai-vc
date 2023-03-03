#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import omegaconf


def get_model_tag(metadata: omegaconf.DictConfig):
    if isinstance(metadata.data, omegaconf.ListConfig):
        data = "_".join(sorted(metadata.data))
    else:
        data = metadata.data

    comment = ""
    if "comment" in metadata:
        comment = f"_{metadata.comment}"

    return f"{metadata.algo}_{metadata.model}_{data}{comment}"
