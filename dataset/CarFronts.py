import os
from PIL import Image
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class FrontDataset(Dataset):
    def __init__(self, csv_path, dataset_path, transform=None, data=None):
        self.csv_path = csv_path
        self.dataset_path = dataset_path
        self.transform = transform

        self.data = self.load_data() if data is None else data

    def load_data(self):
        print(f"Loading file {self.csv_path}")
        assert os.path.exists(
            self.csv_path),  f"Cannot access data file {self.csv_path}"
        df = pd.read_csv(self.csv_path)
        return df.to_numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        car_path, prod_year = self.data[idx][0], self.data[idx][1]
        car_full_path = os.path.join(self.dataset_path + car_path)
        car_image = self.load_image(car_full_path)
        car_image = car_image.convert("RGB")

        if self.transform is not None:
            car_tensor = self.transform(car_image)

        return car_tensor, prod_year

    def load_image(self, path):
        return Image.open(path)

    def print_info(self):
        print(f"Dataset contains {len(self.data)} cars")

    def minimum_prod_year(self):
        return np.amin(self.data, 0)[1]

    def maximum_prod_year(self):
        return np.amax(self.data, 0)[1]

    def shrink_randomly(self, size):
        self.data = self.data[np.random.choice(self.data.shape[0], size, replace=False), :]

    @staticmethod
    def prod_years_range(csv_path):
        assert os.path.exists(
            csv_path),  f"Cannot access data file {csv_path}"
        df = pd.read_csv(csv_path)
        MAXIMUM_YEAR = df.max()['prod_year']
        MINIMUM_YEAR = df.min()['prod_year']

        return MAXIMUM_YEAR - MINIMUM_YEAR + 1


if __name__ == '__main__':
    # Give absolute path to folders
    csv_path = '../processed_fronts.csv'
    dataset_path = "../data/"

    transform = transforms.ToTensor()
    dataset = FrontDataset(csv_path, dataset_path, transform)
    tensor, prod_year = dataset[6]
