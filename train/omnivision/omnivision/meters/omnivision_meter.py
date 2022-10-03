import copy
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Optional, Sequence

import torch
import torch.distributed
from omnivision.utils.generic import recursive_to

from omnivore.utils.distributed import all_gather_batch
from torch.distributed import ReduceOp


class OmnivisionMeter(ABC):
    """Omnivision meter.

    Every deriving meter should call super().__init__() right at the beginning.
    """

    def __init__(self) -> None:
        self._state = {}
        self._default_state = {}
        self._state_reduce_op = {}
        self._sync_device = None

    def register_buffer(
        self, name: str, value: torch.Tensor, reduce_op: Optional[ReduceOp]
    ):
        """Register a buffer for meter computation with a reduction op.

        Args:
            name: Name of the buffer
            value: Initial value, a tensor or a list of tensors
            reduce_op: The reduction to apply during all reduce if the value is a
                tensor. Otherwise, this should be `None` and the sync will do
                an all gather of all the lists and flatten them.
        """
        assert isinstance(value, (torch.Tensor, Sequence))
        if isinstance(value, torch.Tensor):
            assert value.device == torch.device("cpu")
        else:
            assert all(isinstance(v, torch.Tensor) for v in value)
        if isinstance(value, Sequence):
            assert reduce_op is None
            for el in value:
                assert isinstance(
                    el, torch.Tensor
                ), "Arbitary lists not supported yet, due to all_gather_object memory leak"
        else:
            # FIXME: this check crashes on Pytorch 1.13
            # assert isinstance(reduce_op, ReduceOp)
            pass
        self._state[name] = value
        self._state_reduce_op[name] = reduce_op
        # default state assumes register_buffer is only called in __init__
        self._default_state = copy.deepcopy(self._state)

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Update the state of the meter."""
        pass

    @abstractmethod
    def compute(self) -> Dict[str, Any]:
        """Compute the value of the meter.
        Returns a dict to allow for multiple metrics to be
        computed in the same meter (eg top-1, top-5 etc).
        """
        pass

    def set_sync_device(self, device: torch.device) -> Any:
        self._sync_device = device

    @contextmanager
    def sync_state(self) -> None:
        """Context manager to replace the buffers of the meters with synced values."""
        assert self._sync_device is not None
        try:
            orig_state = self._state
            synced_state = recursive_to(
                copy.deepcopy(orig_state), device=self._sync_device
            )
            for name, value in synced_state.items():
                reduce_op = self._state_reduce_op[name]
                if reduce_op is None:
                    value = all_gather_batch(value)
                    # rgirdhar: Not using the following (which technically should
                    # just gather everything in the list) because it seems to have a
                    # memory leak -- it creates processes on each GPU to move the object
                    # and those don't get deleted once the meter is reset.
                    # Discussing in omnivore chat:
                    # fb.workplace.com/chat/t/5260167217327672/?mid=mid.%24gABKwGB5yrjiJHkOMmWC7P47gkCJE
                    # out_vals = [None] * torch.distributed.get_world_size()
                    # torch.distributed.all_gather_object(out_vals, value)
                    # torch.distributed.all_gather_object(out_vals, value)
                    # value = [v for out_val in out_vals for v in out_val]
                else:
                    torch.distributed.all_reduce(value, op=reduce_op)
                synced_state[name] = value
            synced_state = recursive_to(synced_state, device=torch.device("cpu"))
            self._state = synced_state
            yield
        finally:
            self._state = orig_state

    def compute_synced(self) -> Dict[str, Any]:
        """Compute the value of the meter using the synced state.

        Does not change the internal state of the meter after returning
        since it uses a context manager to restore the unsynced state.
        """
        with self.sync_state():
            ret = self.compute()
        return ret

    def __getattr__(self, attr) -> Any:
        # we don't use self._state since when deepcopying
        # self._state is not defined, which results in an infinite loop
        if attr in self.__getattribute__("_state"):
            return self._state[attr]
        raise AttributeError(attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        if attr != "_state" and attr in self._state:
            self._state[attr] = value
        else:
            super().__setattr__(attr, value)

    def reset(self) -> None:
        """Reset the state of the meter to the initial buffer values."""
        self._state = copy.deepcopy(self._default_state)
