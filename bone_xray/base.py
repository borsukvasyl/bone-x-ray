from abc import ABC, abstractmethod
from typing import Union

import albumentations as albu
import numpy as np
import torch
from pytorch_toolbelt.utils import image_to_tensor

from bone_xray.models.cam import ScoreCAM


def to_device(x: Union[torch.Tensor, torch.nn.Module], cuda_id: int = 0):
    return x.cuda(cuda_id) if torch.cuda.is_available() else x


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


class BaseClassificationPredictor(BasePredictor):
    def _process(self, x: torch.Tensor):
        with torch.no_grad():
            x = to_device(x)
            pred = self.model(x).softmax(1).cpu()
            pred = pred[0, 1]  # probability of positive label
        return pred


class BaseLocalizationPredictor(BasePredictor):
    def __init__(self, model: torch.nn.Module, cam_layer: torch.nn.Module, img_size: int):
        model = ScoreCAM(model, cam_layer)
        super().__init__(model, img_size)

    def _process(self, x: torch.Tensor):
        with torch.no_grad():
            x = to_device(x)
            cam, _ = self.model(x)
        return cam.squeeze().numpy()
