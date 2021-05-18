from typing import Dict, Any

from torch import nn

from bone_xray.models.backbones import get_backbone


class Classifier(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.num_classes = config["num_classes"]

        self.backbone, num_channels = get_backbone(config["backbone"], config.get("pretrained", True))
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        if config.get("head", "simple") == "two_layer":
            self.output = nn.Sequential(
                nn.Linear(
                    in_features=num_channels,
                    out_features=512
                ),
                nn.ReLU(inplace=True),
                nn.Linear(
                    in_features=512,
                    out_features=self.num_classes
                ),
            )
        else:
            self.output = nn.Linear(
                in_features=num_channels,
                out_features=self.num_classes
            )

    def forward(self, x):
        x = self.backbone(x)
        x = self.final_pool(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x
