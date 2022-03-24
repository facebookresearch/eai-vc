from glob import glob

from PIL import Image
# There seems to be a corrupt file in the dataset.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torchvision import transforms
from torch.utils.data import Dataset

class OmniDataset(Dataset):
    def __init__(
            self,
            data_root,
            transform,
            mode="train",
            num_points=None,
            datasets="all",
            data_type="14_5m",
    ):
        super().__init__()
        self.data_root = data_root
        self.data_type = data_type
        self.transform = transform
        self.mode = mode
        self.data = self._load_text_files(datasets)
        self.num_dataset_points = num_points or len(self.data)
        self.meta_data = self.data[:self.num_dataset_points]
    
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
        item = Image.open(path).convert('RGB')

        item = self.transform(item)

        return item, 0

    def __len__(self):
        return self.num_dataset_points

if __name__ == '__main__':
    dataset = OmniDataset(
        data_root="/checkpoint/karmeshyadav/omnidataset",
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(224)
        ]),
        datasets="hm3d",
    )
    print("Dataset Len: {}".format(dataset.__len__()))
    n = dataset.__len__()
