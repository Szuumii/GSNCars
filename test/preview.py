from math import prod
import os
import pytorch_lightning as pl
from torch.utils import data
from model.models import DVMModel
from dataset.CarFronts import FrontDataset
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import torch
import torch.utils.data as data_utils


def tensor_to_image(tensor_image):
    img_size = (400, 400)
    to_pil_image = transforms.Compose(
        [transforms.Resize(img_size), transforms.ToPILImage()])
    image = to_pil_image(tensor_image)
    return image


def display_batch(batch):
    imgs, prod_year = batch
    n_images_per_row = 2
    n_rows = int(np.ceil(len(imgs) / n_images_per_row))
    fig, axis = plt.subplots(n_rows, n_images_per_row)
    for ndx in range(len(imgs)):
        row = ndx // n_images_per_row
        col = ndx % n_images_per_row
        tensor = tensor_to_image(imgs[ndx])
        axis[row, col].imshow(tensor)
        axis[row, col].set_xlabel(prod_year[ndx].item())

def display_batch_with_predictions(model, batch):
    imgs, prod_year = batch
    predictions = model.forward(imgs)
    predicted_class = predictions.argmax(dim=1)

    n_images_per_row = 2
    n_rows = int(np.ceil(len(imgs) / n_images_per_row))
    fig, axis = plt.subplots(n_rows, n_images_per_row)
    for ndx in range(len(imgs)):
        row = ndx // n_images_per_row
        col = ndx % n_images_per_row
        tensor = tensor_to_image(imgs[ndx])
        axis[row, col].imshow(tensor)
        axis[row, col].set_xlabel(f"Prod_year: {prod_year[ndx].item()} vs predicted: {2018 - predicted_class[ndx].item()}")
    
    plt.show()

if __name__ == '__main__':
    dataset_root = "/home/shades/Datasets/resized_DVM/"
    ckp_path = "/home/shades/GitRepos/GSNCars/lightning_logs/logs/version_36/checkpoints/epoch=4-step=59974.ckpt"
    csv_path = '/home/shades/GitRepos/GSNCars/csv_files/angle_0/relevant_angle_test_0.csv'
    batch_size = 4

    model = DVMModel.load_from_checkpoint(ckp_path)

    transform = transforms.ToTensor()
    dataset = FrontDataset(csv_path, dataset_root, transform)
    data_loader = DataLoader(dataset, batch_size, shuffle=True)

    batch = next(iter(data_loader))

    imgs, prod_year = batch

    display_batch_with_predictions(model, batch)