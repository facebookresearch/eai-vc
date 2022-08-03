from glob import glob

from PIL import Image
# There seems to be a corrupt file in the dataset.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from typing import List, Optional
import torchvision.transforms.functional as TF

from torchvision import transforms
from torch.utils.data import Dataset

class OmniDataset(Dataset):
    def __init__(
            self,
            data_root,
            transform: Optional[str] = None,
            extra_transform: Optional[str] = None,
            mean: Optional[List[float]] = None,
            std: Optional[List[float]] = None,
            mode: Optional[str] = "train",
            datasets: Optional[str] = "all",
            data_type: Optional[str] = "14_5m",
    ):
        super().__init__()
        self.data_root = data_root
        self.data_type = data_type
        self.transform = transform
        self.mode = mode
        self.meta_data = self._load_text_files(datasets)

        self.extra_transform = extra_transform
        self.mean = mean
        self.std = std
        assert (mean is None) == (std is None)
    
    def _load_text_files(self, datasets):
        data = []

        datasets = datasets.split("-")
        for file_path in glob(f'{self.data_root}/{self.data_type}/*.txt'):
            if "all" not in datasets:
                # datasets is of type "taskonomy-hm3d"
                if file_path.split("/")[-1].split(".")[0] not in datasets:
                    continue
            print(file_path)
            with open(file_path, 'rt') as file:
                scene_dataset = file.readlines()

            data += scene_dataset

        return data

    def __getitem__(self, index):
        meta = self.meta_data[index]

        path = meta.strip()
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        if self.extra_transform is not None:
            extra_img = self.extra_transform(img)
        else:
            extra_img = img.copy()

        img, extra_img = TF.to_tensor(img), TF.to_tensor(extra_img)
        if self.mean is not None and self.std is not None:
            img = TF.normalize(img, self.mean, self.std)
            extra_img = TF.normalize(extra_img, self.mean, self.std)

        return img, extra_img, 0

    def __len__(self):
        return len(self.meta_data)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision.transforms import (
        ColorJitter,
        Compose,
        RandomApply,
        RandomHorizontalFlip,
        RandomResizedCrop,
    )

    dataset = OmniDataset(
        data_root="/checkpoint/karmeshyadav/omnidataset",
        transform=Compose(
            [
                RandomResizedCrop(224, (0.2, 1.0), interpolation=3),
                RandomHorizontalFlip(),
            ]
        ),
        extra_transform=RandomApply(
            [ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8
        ),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        datasets="all",
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    for imgs, extra_imgs, _ in loader:
        print(imgs.shape, extra_imgs.shape)
        break
    print("Dataset Len: {}".format(dataset.__len__()))
    n = dataset.__len__()
