from typing import Dict, Any

from torch import nn

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCEWithLogitsLoss,
}


def get_loss(loss_config: Dict[str, Any]):
    loss_name = loss_config.pop("name")
    return _LOSSES[loss_name](**loss_config)
