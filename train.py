import lightning as L
from lightning import Trainer
import yaml

from dataset import h5Dataset
from torch.utils.data import DataLoader
from model import LightningModule


def train():
    config = yaml.safe_load(open("config.yaml"))
    model = LightningModule(**config['model'])
    dataset = h5Dataset(**config['dataset'])
    trainer = Trainer(limit_train_batches=100, **config['trainer'])

    pass


if __name__ == "__main__":
    train()
