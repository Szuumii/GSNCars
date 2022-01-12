import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from model.models import DVMModel
from dataset.CarFronts import FrontDataset
from torch.utils.data import DataLoader


def tensor_to_image(tensor_image):
    IMG_SIZE = (300, 300)
    to_pil_image = transforms.Compose(
        [transforms.Resize(IMG_SIZE), transforms.ToPILImage()])
    image = to_pil_image(tensor_image)
    return image

def display_car(x, predicted_prod_year, prod_year):
    plt.plot(tensor_to_image(x))
    plt.xlabel(f'Predicted year: {predicted_prod_year} vs production year{prod_year}')
    plt.show()

if __name__ == '__main__':
    #Loading model
    IMG_SIZE = (300, 300)
    test_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Resize(IMG_SIZE)])

    dataset_root = "/home/shades/Datasets/"
    

    test_csv_path = "/home/shades/GitRepos/GSNCars/csv_files/confirmed_fronts/confirmed_front_test.csv"
    ckp_path = "/home/shades/GitRepos/GSNCars/lightning_logs/logs/version_27/checkpoints/epoch=7-step=24735.ckpt"
    model = DVMModel.load_from_checkpoint(ckp_path)
    test_ds = FrontDataset(test_csv_path, dataset_root, test_transform)

    x, label = test_ds[21]

    test_dl = DataLoader(test_ds, batch_size=4)

    for idx, elem in enumerate(test_dl):
        _ , prod_year = model(torch.tensor(elem))


        if idx > 0:
            break


    # _ , prod_year = model.forward(batch)

    # pritn(prod_year)
