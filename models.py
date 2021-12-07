from torchvision import models, transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
from dataset.CarFronts import FrontDataset
from torch import nn

class DVMModel(pl.LightningModule):
    def __init__(self, dataset_csv_path, dataset_dir_path, batch_size=32):
        super().__init__()
        self.dataset_csv_path = dataset_csv_path
        self.dataset_dir_path = dataset_dir_path
        self.batch_size = batch_size

        self.net = models.vgg16(pretrained=True)
        last_in_features = self.net.classifier[6].in_features
        self.net.classifier[6] = nn.Linear(last_in_features, 19)

        self.loss = nn.NLLLoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, _):
        x, _ = batch
        embeddings = self.forward(x)
        loss = self.loss(embeddings)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def prepare_data(self):
        transform = transforms.ToTensor()
        self.dataset = FrontDataset(self.dataset_csv_path, self.dataset_dir_path, transform)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

if __name__ == '__main__':
    csv_path = 'processed_fronts.csv'
    dataset_path = "data/"

    trainer = pl.Trainer(max_epochs=1)
    model = DVMModel(csv_path, dataset_path)
    trainer.fit(model)
    print('hello')
