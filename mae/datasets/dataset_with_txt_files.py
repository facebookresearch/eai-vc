from glob import glob
import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

PLACES_ROOMS = [
    "classroom", "mansion", "patio", "airport_terminal", "beauty_salon", "closet", "dorm_room", "home_office", \
    "bedroom", "engine_room", "hospital_room", "martial_arts_gym", "shed", "cockpit", "hotel_outdoor", "apartment_building_outdoor", \
    "bookstore", "coffee_shop", "hotel_room", "shopfront", "conference_center", "shower", "conference_room", "motel", "pulpit", "fire_escape", \
    "art_gallery", "art_studio", "corridor", "museum_indoor", "railroad_track", "inn_outdoor", "music_studio", "attic", "nursery", "auditorium", \
    "residential_neighborhood", "cafeteria", "office", "restaurant", "waiting_room", "office_building", "restaurant_kitchen", "stage_indoor", \
    "ballroom", "game_room", "kitchen", "restaurant_patio", "staircase", "banquet_hall", "bar", "dinette_home", "living_room", \
    "swimming_pool_outdoor", "basement", "dining_room", "lobby", "parlor", "locker_room"
]

PLACES_OUTDOORS = [
    'abbey', 'alley', 'amphitheater', 'amusement_park', 'aqueduct', 'arch', 'apartment_building_outdoor', 'badlands', \
    'bamboo_forest', 'baseball_field', 'basilica', 'bayou', 'boardwalk', 'boat_deck', 'botanical_garden', 'bridge', 'building_facade', \
    'butte', 'campsite', 'canyon', 'castle', 'cemetery', 'chalet', 'coast', 'construction_site', 'corn_field', 'cottage_garden', 'courthouse', \
    'courtyard', 'creek', 'crevasse', 'crosswalk', 'cathedral_outdoor', 'church_outdoor', 'dam', 'dock', 'driveway', 'desert_sand', \
    'desert_vegetation', 'doorway_outdoor', 'excavation', 'fairway', 'fire_escape', 'fire_station', 'forest_path', 'forest_road', 'formal_garden', \
    'fountain', 'field_cultivated', 'field_wild', 'garbage_dump', 'gas_station', 'golf_course', 'harbor', 'herb_garden', 'highway', 'hospital', \
    'hot_spring', 'hotel_outdoor', 'iceberg', 'igloo', 'islet', 'ice_skating_rink_outdoor', 'inn_outdoor', 'kasbah', 'lighthouse', 'mansion', 'marsh', \
    'mausoleum', 'medina', 'motel', 'mountain', 'mountain_snowy', 'market_outdoor', 'monastery_outdoor', 'ocean', 'office_building', 'orchard', \
    'pagoda', 'palace', 'parking_lot', 'pasture', 'patio', 'pavilion', 'phone_booth', 'picnic_area', 'playground', 'plaza', 'pond', 'racecourse', \
    'raft', 'railroad_track', 'rainforest', 'residential_neighborhood', 'restaurant_patio', 'rice_paddy', 'river', 'rock_arch', 'rope_bridge', 'ruin', \
    'runway', 'sandbar', 'schoolhouse', 'sea_cliff', 'shed', 'shopfront', 'ski_resort', 'ski_slope', 'sky', 'skyscraper', 'slum', 'snowfield', 'swamp', \
    'stadium_baseball', 'stadium_football', 'swimming_pool_outdoor', 'television_studio', 'topiary_garden', 'tower', 'train_railway', 'tree_farm', \
    'trench', 'temple_east_asia', 'temple_south_asia', 'track_outdoor', 'underwater_coral_reef', 'valley', 'vegetable_garden', 'veranda', 'viaduct', \
    'volcano', 'waiting_room', 'water_tower', 'watering_hole', 'wheat_field', 'wind_farm', 'windmill', 'yard'
]

class DatasetWithTxtFiles(Dataset):
    def __init__(
        self,
        data_root,
        transform,
        mode="train",
        dataset_type='full',
        return_index=False
    ):
        self.data_root = data_root
        self.mode = mode
        self.return_index = return_index
        self.transform = transform

        if dataset_type == 'full':
            base_path = os.path.join(self.data_root, 'classes.txt')
            with open(base_path, 'rt') as file:
                folder_names = file.readlines()
                self.places_names = [name.strip() for name in folder_names]
        elif dataset_type == 'rooms':
            self.places_names = PLACES_ROOMS
        elif dataset_type == 'outdoors':
            self.places_names = PLACES_OUTDOORS
        else:
            raise Exception("No dataset named {}".format(dataset_type))

        self.scenes = []
        self.samples = []
        self._load_text_files()

    def _load_text_files(self):
        counter = 0
        data = []

        for file_path in glob(f'{self.data_root}/text_files_{self.mode}/*.txt'):
            file_name = file_path.split('/')[-1][:-4]
            if file_name in self.places_names:
                with open(file_path, 'rt') as file:
                    scene_dataset = file.readlines()

                self.scenes.append(file_name)
                self.samples.extend([(path, counter) for path in scene_dataset])
                counter = counter + 1

        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        im = Image.open(path.strip()).convert('RGB')

        im = self.transform(im)

        if self.return_index:
            return im, index
        else:
            return im, label


if __name__ == "__main__":
    dataset = DatasetWithTxtFiles(
        data_root="/checkpoint/karmeshyadav/places205",
        transform=transforms.RandomHorizontalFlip(p=0.5),
        dataset_type='full')

    print(len(dataset.places_names))

    idx = 4250 * 102

    im, label = dataset[idx]
    print(label, " ", dataset.scenes[label])

    dataset = DatasetWithTxtFiles(
        data_root="/checkpoint/karmeshyadav/imagenet_full_size",
        transform=transforms.RandomHorizontalFlip(p=0.5),
        dataset_type='full')

    print(len(dataset.places_names))

    idx = 4250 * 102

    im, label = dataset[idx]
    print(label, " ", dataset.scenes[label])