import os
import pytorch_lightning as pl
from model.models import DVMModel
import torch

if __name__ == '__main__':
    dataset_root = "/home/shades/Datasets/resized_DVM/"
    logs_path = "/home/shades/GitRepos/GSNCars/lightning_logs"
    ckp_path = "/home/shades/GitRepos/GSNCars/lightning_logs/logs/version_32/checkpoints/epoch=7-step=95959.ckpt"
    batch_size = 16

    torch.cuda.empty_cache()

    tb_logger = pl.loggers.TensorBoardLogger(logs_path, name="logs")

    trainer = pl.Trainer(precision=16, max_epochs=7, gpus=1, logger=tb_logger)
    model = DVMModel(dataset_root, batch_size=batch_size, learning_rate=9e-5)
    # model = DVMModel.load_from_checkpoint(ckp_path)
    trainer.fit(model)