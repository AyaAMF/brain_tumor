import torch
import torch.nn as nn
from torchvision import transforms


IMG_SIZE = 128


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BrainTumorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 63 * 63, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))


def get_train_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])


def get_inference_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])


def load_model(path, device=None):
    if device is None:
        device = get_device()
    m = BrainTumorModel().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m
