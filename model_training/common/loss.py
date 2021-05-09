from typing import Dict, Any

import torch
from torch import nn

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCEWithLogitsLoss,
}


def get_loss(loss_config: Dict[str, Any]):
    loss_name = loss_config.pop("name")
    weights = None
    if "weights" in loss_config:
        weights = loss_config.pop("weights")
        weights = torch.tensor(weights)
    return _LOSSES[loss_name](weight=weights, **loss_config)
