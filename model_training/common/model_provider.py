from typing import Dict, Any

from model_training.common.models.classifier import Classifier

_MODELS = {
    "classifier": Classifier,
}


def get_model(model_name: str, model_config: Dict[str, Any]):
    model = _MODELS[model_name](model_config)
    return model
