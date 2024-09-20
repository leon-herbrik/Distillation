import lightning as L
import torch

from MaskGIT_pytorch.Network import GroundingTransformer
from MaskGIT_pytorch.Network.Taming.models.vqgan import VQModel


class LightningModule(L.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.model = GroundingTransformer(**config)
        self.loss = self._get_loss(config['loss'])
        self.lr = config['lr']

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def _get_loss(self, name):
        if name == 'mse':
            return torch.nn.MSELoss()
        elif name == 'mae':
            return torch.nn.L1Loss()
        else:
            raise ValueError(f"Loss {name} not supported.")
