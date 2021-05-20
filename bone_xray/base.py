from abc import ABC, abstractmethod
from typing import Union

import albumentations as albu
import numpy as np
import torch
from pytorch_toolbelt.utils import image_to_tensor

from bone_xray.utils import get_relative_path


def to_device(x: Union[torch.Tensor, torch.nn.Module], cuda_id: int = 0):
    return x.cuda(cuda_id) if torch.cuda.is_available() else x


def get_default_model():
    config = {
        "img_size": 384,
        "model": {
            "name": "classifier",
            "backbone": "densenet121",
            "num_classes": 2,
            "head": "simple",
            "pretrained": False,
        }
    }
    weights_path = get_relative_path("weights/densenet121.ckpt", __file__)
    return config, weights_path


class BasePredictor(ABC):
    def __init__(
            self,
            model: torch.nn.Module,
            img_size: int,
            mean: Union[float, np.ndarray] = 0.5,
            std: Union[float, np.ndarray] = 0.5
    ):
        self.model = to_device(model.eval())
        self.transform = albu.Compose([
            albu.Resize(img_size, img_size),
            albu.Normalize(mean=mean, std=std)
        ])

    def __call__(self, img: np.ndarray):
        x = self._preprocess(img)
        return self._process(x)

    def _preprocess(self, img: np.ndarray):
        img = self.transform(image=img)["image"]
        x = image_to_tensor(img)[None, ...]
        return x

    @abstractmethod
    def _process(self, x: torch.Tensor):
        pass
