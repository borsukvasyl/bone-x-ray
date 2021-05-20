import cv2
import torch
import yaml
from skimage.io import imread

from bone_xray.classifier import ClassificationPredictor


def test_classifier():
    config = yaml.load("""
        img_size: 384
        model:
            name: classifier
            backbone: densenet121
            num_classes: 2
            head: simple
            pretrained: False
    """)
    CHECKPOINT_PATH = 'tests/files/epoch=26-step=124226.ckpt'
    classifier = ClassificationPredictor(config, CHECKPOINT_PATH)
    img = imread('tests/fixtures/image1.png')
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    pred = classifier(img)
    assert pred.dtype == torch.float32
