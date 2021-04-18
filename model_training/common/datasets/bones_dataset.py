import cv2
import numpy as np
import pandas as pd
from pytorch_toolbelt.utils import image_to_tensor
from skimage.io import imread
from torch.utils.data import Dataset


class BonesDataset(Dataset):
    def __init__(self, path: str, img_size: int, split: str):
        self.data = pd.read_csv(path)
        self.data = self.data[~self.data["label"].isna()]
        self.data = self.data[self.data["train_test"] == split]
        self.img_size = img_size

    def __getitem__(self, i):
        row = self.data.iloc[i]
        img_name = row["path_to_file"]
        label = row["label"]

        try:
            image = imread(img_name)
        except Exception as e:
            print(e)
            return self[0]

        image = cv2.resize(image, (self.img_size, self.img_size))
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = (image / 255.).astype(np.float32)
        image = image_to_tensor(image)

        label = np.array([float(label)])

        return image, label

    def __len__(self):
        return len(self.data)

    @classmethod
    def from_config(cls, config):
        return cls(config["data_path"], config["img_size"], config["split"])
