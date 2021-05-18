from fire import Fire

from model_training.classification.data import BonesDataModule
from model_training.classification.model import ClassificationLightningModel
from model_training.common.trainer import get_trainer
from bone_xray.utils import load_yaml
from bone_xray.models import get_model


def train_classifier(config):
    model_config = config["model"]
    model = get_model(model_config["name"], model_config)
    pl_model = ClassificationLightningModel(model=model, config=config)
    pl_dataset = BonesDataModule(config=config)
    trainer = get_trainer(config)
    trainer.fit(pl_model, pl_dataset)


def main(config_path: str):
    config = load_yaml(config_path)
    train_classifier(config)


if __name__ == '__main__':
    Fire(main)
