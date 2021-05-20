import cv2
import torch
from skimage.io import imread

from bone_xray.classifier import Densenet121ClassificationPredictor


def test_classifier():
    classifier = Densenet121ClassificationPredictor()
    img = imread('tests/fixtures/image1.png')
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    pred = classifier(img)
    assert pred.dtype == torch.float32
