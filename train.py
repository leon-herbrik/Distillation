import os

import yaml
import lightning as L
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from dataset.dataset import h5Dataset
from model.lightning_module import LightningModule


def train():
    config = yaml.safe_load(open("config.yaml"))
    model = LightningModule(config['model'])
    train_dataset = h5Dataset(**config['dataset'], split='train')
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config['trainer']['batch_size'],
                                  shuffle=True,
                                  num_workers=128)
    val_dataset = h5Dataset(**config['dataset'], split='val')
    val_dataloader = DataLoader(val_dataset,
                                batch_size=config['trainer']['batch_size'],
                                shuffle=False,
                                num_workers=128)
    logger = WandbLogger(**config['logger'])
    trainer = Trainer(**config['trainer']['pl_trainer'], logger=logger)
    trainer.fit(model, train_dataloader, val_dataloader)

    pass


if __name__ == "__main__":
    train()
