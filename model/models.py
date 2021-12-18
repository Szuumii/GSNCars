from torchvision import models, transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
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
        self.PROD_YEAR_RANGE = 18

        self.MAX_PROD_YEAR = 2018

        self.net = models.vgg16(pretrained=True)
        last_in_features = self.net.classifier[6].in_features
        self.net.classifier[6] = nn.Linear(last_in_features, self.PROD_YEAR_RANGE)

        self.loss = nn.CrossEntropyLoss()
        # self.metric = pl.metrics.MeanAbsoluteError()

    def forward(self, x):
        return self.net(x)

    def prepare_data(self):
        SMALL_DATA_PERCENTAGE = 0.2
        IMG_SIZE = (290, 290)

        train_csv_path = "/home/shades/GitRepos/GSNCars/csv_files/angle_0/relevant_angle_train_0.csv"
        val_csv_path = "/home/shades/GitRepos/GSNCars/csv_files/angle_0/relevant_angle_val_0.csv"
        test_csv_path = "/home/shades/GitRepos/GSNCars/csv_files/angle_0/relevant_angle_test_0.csv"

        train_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Resize(IMG_SIZE)])

        self.train_dataset = FrontDataset(train_csv_path, self.dataset_dir_path, train_transform)

        val_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Resize(IMG_SIZE)])

        self.validation_dataset = FrontDataset(val_csv_path, self.dataset_dir_path, val_transform)

        test_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Resize(IMG_SIZE)])

        self.test_dataset = FrontDataset(test_csv_path, self.dataset_dir_path, test_transform)

        if self.small_train:
            self.train_dataset.shrink_randomly(SMALL_DATA_PERCENTAGE)
            self.validation_dataset.shrink_randomly(SMALL_DATA_PERCENTAGE)
            self.test_dataset.shrink_randomly(SMALL_DATA_PERCENTAGE)

    def training_step(self, batch, batch_idx):
        x, prod_year = batch
        predictions = self.forward(x)
        target = self.target_for_prod_year(prod_year)
        loss = self.loss(predictions, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, prod_year = batch
        predictions = self.forward(x)
        target = self.target_for_prod_year(prod_year)
        loss = self.loss(predictions, target)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    # def test_step(self, batch, batch_idx):
    #     loss, acc = self._shared_eval_step(batch, batch_idx)
    #     metrics = {"test_acc": acc, "test_loss": loss}
    #     self.log_dict(metrics)
    #     return metrics

    # def _shared_eval_step(self, batch, batch_idx):
    #     x, prod_year = batch
    #     predictions = self.model(x)
    #     loss = self.loss(y_hat, y)
    #     mea = self.metric(y_hat, prod_year)
    #     return loss, acc

    def target_for_prod_year(model, prod_year):
        targets = torch.zeros((prod_year.shape[0], model.PROD_YEAR_RANGE), device="cuda")
        for target, yr in zip(targets, prod_year):
            target[model.MAX_PROD_YEAR - yr] = 1
        return targets

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
