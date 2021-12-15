from torchvision import models, transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import torch
from dataset.CarFronts import FrontDataset
from torch import nn

class DVMModel(pl.LightningModule):
    def __init__(self, dataset_dir_path, batch_size=32, learning_rate=1e-6, small_train=False):
        super().__init__()
        self.dataset_dir_path = dataset_dir_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.small_train = small_train

        # Prodyear range predetermined upfront based on the dataset
        PROD_YEAR_RANGE = 21

        self.net = models.vgg16(pretrained=True)
        last_in_features = self.net.classifier[6].in_features
        self.net.classifier[6] = nn.Linear(last_in_features, PROD_YEAR_RANGE)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def prepare_data(self):
        SMALL_DATA_PERCENTAGE = 0.2

        train_csv_path = ""
        val_csv_path = ""
        test_csv_path = ""

        train_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Resize(300, 300)])

        self.train_dataset = FrontDataset(train_csv_path, self.dataset_dir_path, train_transform)

        val_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Resize(300, 300)])

        self.validation_dataset = FrontDataset(val_csv_path, self.dataset_dir_path, val_transform)

        test_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Resize(300, 300)])

        self.test_dataset = FrontDataset(test_csv_path, self.dataset_dir_path, test_transform)
        
        if self.small_train:
            self.train_dataset.shrink_randomly(SMALL_DATA_PERCENTAGE)
            self.validation_dataset.shrink_randomly(SMALL_DATA_PERCENTAGE)
            self.test_dataset.shrink_randomly(SMALL_DATA_PERCENTAGE)

    def training_step(self, batch, batch_idx):
        x, prod_year = batch
        predictions = self.forward(x)
        loss = self.loss(predictions, (prod_year - self.train_dataset.minimum_prod_year()))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, prod_year = batch
        predictions = self.forward(x)
        loss = self.loss(predictions, (prod_year - self.train_dataset.minimum_prod_year()))
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

if __name__ == '__main__':
    dataset_path = "data/"

    trainer = pl.Trainer(max_epochs=1)
    model = DVMModel(dataset_path, small_train=True)
    trainer.fit(model)
