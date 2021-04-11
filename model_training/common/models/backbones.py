from pytorchcv.model_provider import get_model
from torch import nn

_BACKBONES = {
    "efficientnet_b0",
    "efficientnet_b0b",
    "efficientnet_b0c",
    "efficientnet_b1",
    "efficientnet_b3b",
    "efficientnet_b5b",
    "efficientnet_b7b",
}

_STAGED_BACKBONES = {
    "densenet161",
    "densenet121",
    "seresnext50_32x4d",
    "seresnext101_32x4d",
}


_NUM_CHANNELS = {
    "efficientnet_b0": 320,
    "efficientnet_b0b": 320,
    "efficientnet_b0c": 320,
    "efficientnet_b1": 320,
    "efficientnet_b3b": 384,
    "efficientnet_b5b": 512,
    "efficientnet_b7b": 640,
    "densenet161": 2208,
    "densenet121": 1024,
    "seresnext50_32x4d": 2048,
    "seresnext101_32x4d": 2048,
}


def get_backbone(backbone_name: str, pretrained: bool = True):
    model = get_model(backbone_name, pretrained=pretrained)
    num_channels = _NUM_CHANNELS[backbone_name]
    if backbone_name in _BACKBONES:
        backbone = nn.Sequential(model.init_block, model.stage1, model.stage2, model.stage3, model.stage4, model.stage5)
    elif backbone_name in _BACKBONES:
        backbone = nn.Sequential(model.init_block, model.stage1, model.stage2, model.stage3, model.stage4)
    else:
        raise ValueError(f"Invalid backbone name [{backbone_name}]")
    return backbone, num_channels
