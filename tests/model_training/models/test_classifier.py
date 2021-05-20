import unittest

import cv2
import torch
from skimage.io import imread

from bone_xray.classifier import Densenet121ClassificationPredictor


class ClassificationTestCase(unittest.TestCase):
    def test_model(self) -> None:
        classifier = Densenet121ClassificationPredictor()
        img = imread('tests/fixtures/image1.png')
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        pred = classifier(img)

        self.assertEqual(pred.dtype, torch.float32)
        self.assertTrue(0 <= pred <= 1)
