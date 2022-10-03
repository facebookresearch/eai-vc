# Lazy import
try:
    from large_vision_dataset.factory import make_dataset
except ImportError:
    raise (
        "Please install the lavida package from https://github.com/fairinternal/lavida"
    )
import random

from omnivore.data.api import VisionTextSample


class LavidaDaasetWrapper:
    def __init__(self, str_or_path):
        self.dataset = make_dataset(str_or_path)

    def __getitem__(self, idx):
        try:
            img, text = self.dataset[idx]
        except Exception as e:
            # FIXME: A hack to workaround some empty archives for Laion-2B on AWS
            print(e)
            return self.__getitem__(random.randint(0, len(self.dataset) - 1))

        text = text.decode("utf-8")
        return VisionTextSample(
            vision=img, data_idx=-1, label=-1, data_valid=True, text=text
        )

    def __len__(self):
        return len(self.dataset)
