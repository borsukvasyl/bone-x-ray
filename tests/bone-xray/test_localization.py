import unittest

import cv2
import numpy as np
from skimage.io import imread

from bone_xray.localization import Densenet121LocalizationPredictor, visualize_heatmap


class LocalizationTestCase(unittest.TestCase):
    def test_model(self) -> None:
        model = Densenet121LocalizationPredictor()
        img = imread('tests/fixtures/image1.png')
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        pred = model(img)

        self.assertIsInstance(pred, np.ndarray)
        self.assertEqual(pred.dtype, np.float32)
        self.assertEqual(len(pred.shape), 2)
        self.assertEqual(pred.shape[0], 384)
        self.assertEqual(pred.shape[1], 384)

    def test_visualization(self) -> None:
        img = imread('tests/fixtures/image1.png')
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cam = np.ones((384, 384), np.float32)
        vis = visualize_heatmap(img, cam)

        self.assertEqual(vis.dtype, np.uint8)
        self.assertEqual(len(vis.shape), 3)
        self.assertEqual(vis.shape, img.shape)
