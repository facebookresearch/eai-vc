from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


class DictDataset(Dataset):
    def __init__(
        self,
        load_data: Dict[str, torch.Tensor],
        load_keys: Optional[List[str]] = None,
        detach_all: bool = True,
    ):
        """
        :parameters load_keys: Subset of keys that are loaded from `load_data`.
        """
        if load_keys is None:
            load_keys = load_data.keys()

        self._load_data = {
            k: v.detach() if detach_all else v
            for k, v in load_data.items()
            if k in load_keys
        }
        tensor_sizes = [tensor.size(0) for tensor in self._load_data.values()]
        if len(set(tensor_sizes)) != 1:
            raise ValueError("Tensors to dataset are not of the same shape")
        self._dataset_len = tensor_sizes[0]

    @property
    def all_data(self):
        return self._load_data

    def get_data(self, k: str) -> torch.Tensor:
        return self._load_data[k]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in self._load_data.items()}

    def __len__(self) -> int:
        return self._dataset_len
