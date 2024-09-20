import os

import yaml
import lightning as L
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner
from torch.utils.data import DataLoader
import torch

from dataset.dataset import h5Dataset
from model.lightning_module import LightningModule
from util.util import are_all_A100


def train():

    # Set matmul precision if all GPUs are A100.
    if are_all_A100():
        torch.set_float32_matmul_precision("medium")

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

    # Start by tuning.
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, train_dataloader, val_dataloader)
    fig = lr_finder.plot(suggest=True)
    fig.savefig("lr_finder.png")

    trainer.fit(model, train_dataloader, val_dataloader)

    pass


if __name__ == "__main__":
    train()
