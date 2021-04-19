import torch
import pytorch_lightning as pl
from pytorch_toolbelt.optimization.functional import get_optimizable_parameters
from torch import optim

from model_training.common.loss import get_loss


class ClassificationLightningModel(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.config = config
        self.model = model
        self.loss = get_loss(config["loss"])
        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)
        loss = self.loss(preds, targets)

        # logging
        self.log('train/loss', loss, on_epoch=True)
        self.log('train/accuracy', self.accuracy(preds.sigmoid(), targets.int()), prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)
        loss = self.loss(preds, targets)

        # logging
        self.log('val/loss', loss, on_epoch=True)
        self.log('val/accuracy', self.accuracy(preds.sigmoid(), targets.int()), on_epoch=True)

    def configure_optimizers(self):
        # get optimizer
        optimizer_config = self.config["optimizer"]
        params = get_optimizable_parameters(self.model)
        optimizer = torch.optim.Adam(params, lr=optimizer_config.get("lr", 1e-4))

        # get scheduler
        scheduler_config = self.config["scheduler"]
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.config.get("metric_mode", "min"),
            patience=scheduler_config["patience"],
            factor=scheduler_config["factor"],
            min_lr=scheduler_config["min_lr"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": scheduler_config.get("metric_to_monitor", "train/loss"),
        }
