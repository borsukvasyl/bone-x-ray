import torch
import pytorch_lightning as pl
from pytorch_toolbelt.optimization.functional import get_optimizable_parameters

from model_training.common.loss import get_loss


class ClassificationLightningModel(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.loss = get_loss(config["loss"])

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)
        loss = self.loss(preds, targets)

        # logging
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        params = get_optimizable_parameters(self.model)
        optimizer = torch.optim.Adam(params, lr=1e-3)
        return optimizer
