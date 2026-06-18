import os

import cv2
import torch
from torch.utils.data import Dataset

from model import get_train_transform


class BrainDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(path):
    transform = get_train_transform()
    X, y = [], []

    for label, folder in enumerate(["no", "yes"]):
        folder_path = os.path.join(path, folder)

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transform(img)

            X.append(img)
            y.append(label)

    return torch.stack(X), torch.tensor(y)
