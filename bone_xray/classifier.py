from typing import Optional

import torch

from bone_xray.base import BaseClassificationPredictor
from bone_xray.utils import get_relative_path


class ClassificationPredictor(BaseClassificationPredictor):
    def __init__(self, checkpoint_path: Optional[str] = None, img_size: int = 384):
        if checkpoint_path is None:
            checkpoint_path = get_relative_path("models/densenet121.trcd", __file__)
        model = torch.jit.load(checkpoint_path)
        super().__init__(model, img_size)
