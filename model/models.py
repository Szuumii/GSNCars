from numpy import angle
from torchvision import models, transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
from dataset.CarFronts import FrontDataset
from torch import nn
from torchmetrics import MeanAbsoluteError

class DVMModel(pl.LightningModule):
    def __init__(self, dataset_dir_path, architecture='VGG16', image_angle=45, csv_path='', batch_size=32, learning_rate=1e-6, small_train=False):
        super().__init__()
        
        self.save_hyperparameters()
        self.dataset_dir_path = dataset_dir_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.small_train = small_train
        self.image_angle = image_angle
        self.csv_path = csv_path

        # Prodyear range predetermined upfront based on the dataset

        self.MAX_PROD_YEAR = 2018
        self.MIN_PROD_YEAR = 2001
        
        self.PROD_YEAR_RANGE = self.MAX_PROD_YEAR - self.MIN_PROD_YEAR + 1

        if architecture is 'VGG16':
            self.net = models.vgg16(pretrained=True)
            last_in_features = self.net.classifier[6].in_features
            self.net.classifier[6] = nn.Linear(last_in_features, self.PROD_YEAR_RANGE)

        elif architecture is 'Resnet50':
            self.net = models.resnet50(pretrained=True)
            self.net.fc = nn.Linear(self.net.fc.in_features, self.PROD_YEAR_RANGE)

        elif architecture is 'EfficientNet':
            self.net = models.efficientnet_b4(pretrained=True)
            last_in_features = self.net.classifier[1].in_features
            self.net.classifier[1] = nn.Linear(last_in_features, self.PROD_YEAR_RANGE)

        else:
            raise Exception('Unknown net architecture! Use VGG16, Resnet50 or EfficientNet')

        self.loss = nn.CrossEntropyLoss()
        self.metric = MeanAbsoluteError()

    def forward(self, x):
        return self.net(x)

    def prepare_data(self):
        SMALL_DATA_PERCENTAGE = 0.4
        IMG_SIZE = (300, 300)

        train_csv_path = f"{self.csv_path}/csv_files/angle_{self.image_angle}/relevant_angle_train_{self.image_angle}.csv"
        val_csv_path = f"{self.csv_path}/csv_files/angle_{self.image_angle}/relevant_angle_val_{self.image_angle}.csv"
        test_csv_path = f"{self.csv_path}/csv_files/angle_{self.image_angle}/relevant_angle_test_{self.image_angle}.csv"

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
        loss, mae = self._shared_eval_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_mea", mae, on_step=False, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        loss, mae = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_mae": mae, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def test_epoch_end(self, outputs) -> None:
        suma = 0
        for out in outputs:
            suma += out['test_mae']

        suma /= len(outputs)
        self.log("test_epoch_mea", suma)

    def _shared_eval_step(self, batch, batch_idx):
        x, prod_year = batch
        predictions = self.forward(x)
        target = self.target_for_prod_year(prod_year)
        predicted_class = predictions.argmax(dim=1)
        target_class = torch.tensor(self.MAX_PROD_YEAR - prod_year.clone(), device="cuda")
        loss = self.loss(predictions, target)
        mea = self.metric(predicted_class, target_class)
        self.metric.reset()
        return loss, mea

    def target_for_prod_year(self, prod_year):
        targets = torch.zeros((prod_year.shape[0], self.PROD_YEAR_RANGE), device="cuda")
        for target, yr in zip(targets, prod_year):
            target[self.MAX_PROD_YEAR - yr] = 1
        return targets

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=24)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=24)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=24)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
