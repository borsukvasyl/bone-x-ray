from typing import Dict, Any, Optional

import torch

from bone_xray.models.classifier import Classifier

_MODELS = {
    "classifier": Classifier,
}


def load_weights(model: torch.nn.Module, model_weights: str, map_location: str = "cpu"):
    weights = torch.load(model_weights, map_location=map_location)
    state_dict = {k.lstrip("model").lstrip("."): v for k, v in weights["state_dict"].items()}
    state_dict.pop("ss.weight", None)
    model.load_state_dict(state_dict)


def get_model(model_name: str, model_config: Dict[str, Any], model_weights: Optional[str] = None):
    model = _MODELS[model_name](model_config)
    if model_weights is not None:
        load_weights(model, model_weights)
    return model
