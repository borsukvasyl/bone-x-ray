import sys

import cv2
import numpy as np
import torch
from pytorch_toolbelt.utils import image_to_tensor

from model_training.classification.predictor import ClassificationPredictor

sys.path.append("venv/src/scorecam")

from cam import ScoreCAM


class LocalizationPredictor(ClassificationPredictor):

    def __call__(self, img):
        inp = self.transform(image=img)["image"]
        inp = image_to_tensor(inp)[None, ...]

        target_layer = self.pl_model.model.backbone[4].unit16.conv2.conv
        wrapped_model = ScoreCAM(self.pl_model, target_layer)

        with torch.no_grad():
            pred = self.pl_model(inp.cuda()).argmax(1).cpu()
            pred = np.squeeze(pred.numpy())
            cam, _ = wrapped_model(inp.cuda())

        return pred, self.show_custom(img, cam.squeeze().numpy())

    @staticmethod
    def show_custom(img: np.ndarray,
                    mask: np.ndarray,
                    use_rgb: bool = False,
                    colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        shape = img.shape[:-1][::-1]
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = cv2.resize(heatmap, dsize=shape, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, dsize=shape, interpolation=cv2.INTER_CUBIC)
        return (img * (1 - mask[..., np.newaxis]) + heatmap * mask[..., np.newaxis]).astype(np.uint8)
