import cv2
import numpy as np
import torch

from bone_xray.base import BaseLocalizationPredictor


def visualize_heatmap(img: np.ndarray,
                      mask: np.ndarray,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    shape = img.shape[:-1][::-1]
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, dsize=shape, interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, dsize=shape, interpolation=cv2.INTER_CUBIC)
    return (img * (1 - mask[..., np.newaxis]) + heatmap * mask[..., np.newaxis]).astype(np.uint8)


class LocalizationPredictor(BaseLocalizationPredictor):
    def __init__(self, checkpoint_path: str, img_size: int = 384):
        model = torch.jit.load(checkpoint_path)
        cam_layer = self.model.model.backbone[4].unit16.conv2.conv
        super().__init__(model, cam_layer, img_size)
