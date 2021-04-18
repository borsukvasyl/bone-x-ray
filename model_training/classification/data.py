from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from model_training.common.datasets import get_dataset


class BonesDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        self.data_train = get_dataset(self.config["train"])
        self.data_val = get_dataset(self.config["val"])

    def train_dataloader(self):
        data_train = DataLoader(self.data_train, batch_size=self.config["batch_size"], num_workers=8, shuffle=True)
        return data_train

    def val_dataloader(self):
        data_val = DataLoader(self.data_val, batch_size=self.config["batch_size"], num_workers=8, shuffle=False)
        return data_val
