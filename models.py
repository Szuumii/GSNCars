from torchvision import models, transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import torch
from dataset.CarFronts import FrontDataset
from torch import nn

class DVMModel(pl.LightningModule):
    def __init__(self, dataset_csv_path, dataset_dir_path, batch_size=32, learning_rate=0.0002, input_size=(300, 300), small_train=False):
        super().__init__()
        self.dataset_csv_path = dataset_csv_path
        self.dataset_dir_path = dataset_dir_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.small_train = small_train
        self.input_size = input_size

        PROD_YEAR_RANGE = FrontDataset.prod_years_range(csv_path)

        self.net = models.vgg16(pretrained=True)
        last_in_features = self.net.classifier[6].in_features
        self.net.classifier[6] = nn.Linear(last_in_features, PROD_YEAR_RANGE)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def prepare_data(self):
        SMALL_DATA_PERCENTAGE = 0.01
        TRAIN_PERCENTAGE = 0.8
        VALIDATION_PERCENTAGE = 0.1

        transform = transforms.Compose([
            transforms.ToTensor(), transforms.Resize(self.input_size)])
        self.train_dataset = FrontDataset(self.dataset_csv_path, self.dataset_dir_path, transform)

        if self.small_train:
            TARGET_SIZE = int(len(self.train_dataset) * SMALL_DATA_PERCENTAGE)
            self.train_dataset.shrink_randomly(TARGET_SIZE)

        data = self.train_dataset.data
        first_split = int(len(data) * TRAIN_PERCENTAGE)
        second_split = int(len(data) * (TRAIN_PERCENTAGE + VALIDATION_PERCENTAGE))
        
        np.random.shuffle(data)
        train_data = data[:first_split]
        test_data = data[first_split:second_split]
        validation_data = data[second_split:]
        
        self.train_dataset.data = train_data
        self.test_dataset = FrontDataset(self.dataset_csv_path, self.dataset_dir_path, transform, test_data)
        self.validation_dataset = FrontDataset(self.dataset_csv_path, self.dataset_dir_path, transform, validation_data)

    def training_step(self, batch, _):
        x, prod_year = batch
        predictions = self.forward(x)
        loss = self.loss(predictions, (prod_year - self.train_dataset.minimum_prod_year()))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8)

if __name__ == '__main__':
    csv_path = 'processed_fronts.csv'
    dataset_path = "data/"

    trainer = pl.Trainer(max_epochs=1)
    model = DVMModel(csv_path, dataset_path, small_train=True)
    trainer.fit(model)
