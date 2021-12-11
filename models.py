from torchvision import models, transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
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
        transform = transforms.Compose([
            transforms.ToTensor(), transforms.Resize(self.input_size)])
        self.dataset = FrontDataset(self.dataset_csv_path, self.dataset_dir_path, transform)

        if self.small_train:
            TARGET_SIZE = int(len(self.dataset) * 0.05) # TODO Arbitrary 5%, how would we like to specify this?
            self.dataset.shrink_randomly(TARGET_SIZE)

    def training_step(self, batch, _):
        x, prod_year = batch
        predictions = self.forward(x)
        loss = self.loss(predictions, (prod_year - self.dataset.minimum_prod_year()))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

if __name__ == '__main__':
    csv_path = 'processed_fronts.csv'
    dataset_path = "data/"

    trainer = pl.Trainer(max_epochs=1)
    model = DVMModel(csv_path, dataset_path, small_train=True)
    trainer.fit(model)
