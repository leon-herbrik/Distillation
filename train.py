import lightning as L
import yaml

from dataset import ptDataset
from torch.utils.data import DataLoader
from model import LightningModule


def train():
    config = yaml.safe_load(open("config.yaml"))

    pass


if __name__ == "__main__":
    train()
