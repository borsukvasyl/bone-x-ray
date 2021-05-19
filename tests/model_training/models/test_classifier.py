import cv2
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from skimage.io import imread
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix

from typing import Dict

from bone_xray.base import BaseClassificationPredictor
from bone_xray.classifier import ClassificationPredictor
from model_training.classification.model import ClassificationLightningModel
from bone_xray.classifier import get_model
from model_training.common.loss import get_loss


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
