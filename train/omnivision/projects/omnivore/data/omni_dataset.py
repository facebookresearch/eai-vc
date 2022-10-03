from abc import ABC, abstractmethod
from typing import Any, Iterable


class OmniDataset(ABC):
    @abstractmethod
    def get_loader(self, *args, **kwargs) -> Iterable:
        pass

    def load_checkpoint_state(self, dataloader_state) -> None:
        return None

    def get_checkpoint_state(self) -> Any:
        return None
