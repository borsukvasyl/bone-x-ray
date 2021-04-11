from fire import Fire

from pl_bolts.datasets import DummyDataset
from torch.utils.data import DataLoader

from model_training.classification.model import ClassificationLightningModel
from model_training.common.trainer import get_trainer
from model_training.common.utils import load_yaml
from model_training.common.model_provider import get_model


def train_classifier(config):
    model_config = config["model"]
    model = get_model(model_config["name"], model_config)

    train_dataset = DummyDataset((3, 256, 256), (1,), num_samples=512)
    train_dataset = DataLoader(train_dataset, batch_size=32)
    val_dataset = DummyDataset((3, 256, 256), (1,), num_samples=512)
    val_dataset = DataLoader(val_dataset, batch_size=32)

    pl_model = ClassificationLightningModel(model=model, config=config)
    trainer = get_trainer(config)
    trainer.fit(pl_model, train_dataset, val_dataset)


def main(config_path: str):
    config = load_yaml(config_path)
    train_classifier(config)


if __name__ == '__main__':
    Fire(main)
