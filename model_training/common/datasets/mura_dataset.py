import os

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
from pytorch_toolbelt.utils import image_to_tensor
from skimage.io import imread
from torch.utils.data import Dataset


class MURADataset(Dataset):
    def __init__(self, path: str, img_size: int, prefix: str):
        self.data = pd.read_csv(path)
        self.img_size = img_size
        self.prefix = prefix
        self.transform = albu.Compose([
            albu.HorizontalFlip(),
            albu.Rotate(),
            albu.Resize(self.img_size, self.img_size),
            albu.Normalize(mean=0.5, std=0.5)
        ])

    def __getitem__(self, i):
        row = self.data.iloc[i]
        image_path = os.path.join(self.prefix, row[0])

        try:
            image = imread(image_path)
        except Exception as e:
            print(e)
            return self[0]

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = self.transform(image=image)["image"]
        image = image_to_tensor(image)

        label = int("_positive/" in image_path)
        label = np.array([float(label)])

        return image, label

    def __len__(self):
        return len(self.data)

    @classmethod
    def from_config(cls, config):
        return cls(config["data_path"], config["img_size"], config["prefix"])
