import torch

from bone_xray.base import BaseClassificationPredictor


class ClassificationPredictor(BaseClassificationPredictor):
    def __init__(self, checkpoint_path: str, img_size: int = 384):
        model = torch.jit.load(checkpoint_path)
        super().__init__(model, img_size)
