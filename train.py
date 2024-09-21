import os
import sys
from functools import partial

import yaml
import lightning as L
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import torch

from dataset.dataset import h5Dataset
from model.lightning_module import LightningModule
from util.util import are_all_A100, model_name


def train():

    # Set matmul precision if all GPUs are A100.
    if are_all_A100():
        torch.set_float32_matmul_precision("medium")

    config_path = sys.argv[1] if len(
        sys.argv) > 1 else "config_bender_1_frame.yaml"

    config = yaml.safe_load(open(config_path))

    model_name = model_name(config)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        filename=model_name + "-{epoch:02d}-{val_loss:.2f}",
        dirpath='./checkpoints',
    )

    model = LightningModule(config['model'])
    vocabulary = model.model.vocabulary
    # Create collate function arguments.
    collate_fn_args = {
        'mask_token': vocabulary['[mask]'],
        'codebook_size': config['model']['codebook_size'],
        'past_shift': vocabulary.past_shift,
        'future_shift': vocabulary.future_shift,
        'feature_token': vocabulary['[feat]'],
        'sep_token': vocabulary['[sep]'],
    }

    train_dataset = h5Dataset(
        **config['dataset'],
        split='train',
        **collate_fn_args,
    )
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config['trainer']['batch_size'],
                                  shuffle=True,
                                  num_workers=128,
                                  collate_fn=train_dataset.collate_fn)
    val_dataset = h5Dataset(**config['dataset'],
                            split='val',
                            **collate_fn_args)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=config['trainer']['batch_size'],
                                shuffle=False,
                                num_workers=128,
                                collate_fn=val_dataset.collate_fn)
    logger = WandbLogger(**config['logger'])
    trainer = Trainer(**config['trainer']['pl_trainer'],
                      logger=logger,
                      callbacks=[checkpoint_callback])

    # # Start by tuning.
    # tuner = Tuner(trainer)
    # lr_finder = tuner.lr_find(model, train_dataloader, val_dataloader)
    # fig = lr_finder.plot(suggest=True)
    # fig.savefig("lr_finder.png")

    trainer.fit(model, train_dataloader, val_dataloader)

    pass


if __name__ == "__main__":
    train()
