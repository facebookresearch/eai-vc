from typing import Optional
import glob
import os

import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset


# class list from:
# https://github.com/yilundu/crl/blob/6adf009d30f292cdc995eb70bab500b0033c11d4/places_finetune/finetune_places.py
INDOOR_CLASSES = [
    "classroom",
    "mansion",
    "patio",
    "airport_terminal",
    "beauty_salon",
    "closet",
    "dorm_room",
    "home_office",
    "bedroom",
    "engine_room",
    "hospital_room",
    "martial_arts_gym",
    "shed",
    "cockpit",
    "hotel-outdoor",
    "apartment_building-outdoor",
    "bookstore",
    "coffee_shop",
    "hotel_room",
    "shopfront",
    "conference_center",
    "shower",
    "conference_room",
    "motel",
    "pulpit",
    "fire_escape",
    "art_gallery",
    "art_studio",
    "corridor",
    "museum-indoor",
    "railroad_track",
    "inn-outdoor",
    "music_studio",
    "attic",
    "nursery",
    "auditorium",
    "residential_neighborhood",
    "cafeteria",
    "office",
    "restaurant",
    "waiting_room",
    "office_building",
    "restaurant_kitchen",
    "stage-indoor",
    "ballroom",
    "game_room",
    "kitchen",
    "restaurant_patio",
    "staircase",
    "banquet_hall",
    "bar",
    "dinette_home",
    "living_room",
    "swimming_pool-outdoor",
    "basement",
    "dining_room",
    "lobby",
    "parlor",
    "locker_room",
]


class PlacesIndoor(VisionDataset):
    def __init__(self, root: str, transform: Optional[str] = None):
        super().__init__(root=root, transform=transform)
        folders = [os.path.join(self.root, c) for c in INDOOR_CLASSES]
        folders = [path for path in folders if os.path.exists(path)]
        self.num_classes = len(folders)
        print("Found {} classes".format(self.num_classes))

        self.files, self.labels = [], []
        for idx, folder in enumerate(folders):
            files = sorted(glob.glob(os.path.join(folder, "*.jpg")))
            self.files.extend(files)
            self.labels.extend([idx] * len(files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        lbl = self.labels[index]
        return np.array(img), lbl


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision.transforms import ToTensor

    dataset = PlacesIndoor(
        root="data/datasets/places365_standard/train", transform=ToTensor()
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    for samples, targets in loader:
        print(samples.size(), targets.size())
        break
