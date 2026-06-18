import torch
import torch.nn as nn


IMG_SIZE = 128


class BrainTumorModel(nn.Module):
    """CNN model for brain tumor binary classification."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 63 * 63, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))
