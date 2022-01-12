import os
import pytorch_lightning as pl
from model.models import DVMModel
import torch

if __name__ == '__main__':
    dataset_root = "/home/shades/Datasets/"
    ckp_path = "/home/shades/GitRepos/GSNCars/lightning_logs/logs/version_34/checkpoints/epoch=3-step=47979.ckpt"
    batch_size = 16

    torch.cuda.empty_cache()

    tb_logger = pl.loggers.TensorBoardLogger("../lightning_logs", name="logs")

    trainer = pl.Trainer(max_epochs=3, gpus=1, logger=tb_logger)
    # model = DVMModel(dataset_root, batch_size=batch_size, small_train=True, learning_rate=6e-5)
    model = DVMModel.load_from_checkpoint(ckp_path)
    trainer.test(model)