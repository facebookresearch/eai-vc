import logging
import os

import torch
import torch.distributed as dist


def init_ranks(pl_trainer):
    assert torch.distributed.is_available() and torch.distributed.is_initialized()

    if os.environ.get("LOCAL_RANK") is not None:
        assert os.environ["LOCAL_RANK"] == str(pl_trainer.local_rank)
    else:
        os.environ["LOCAL_RANK"] = str(pl_trainer.local_rank)
        logging.info(f"Initialized with local rank [{pl_trainer.local_rank}]")

    if os.environ.get("RANK") is not None:
        assert os.environ["RANK"] == str(pl_trainer.global_rank)
    else:
        os.environ["RANK"] = str(pl_trainer.global_rank)
        logging.info(f"Initialized with rank [{pl_trainer.global_rank}]")


def is_local_primary():
    return int(os.getenv("LOCAL_RANK")) == 0


def is_local_primary_cuda():
    assert dist.is_initialized()
    assert torch.cuda.is_available()
    return torch.cuda.current_device() == 0


def is_torch_dataloader_worker():
    return torch.utils.data.get_worker_info() is not None
