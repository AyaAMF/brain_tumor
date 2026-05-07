import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import cv2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DATA_PATH = "brain_tumor_dataset"

IMG_SIZE = 128

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ---------- Load Dataset ----------
def load_data(path):
    X, y = [], []
    for label, folder in enumerate(['no', 'yes']):
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

print("Loading dataset...")
X, y = load_data(DATA_PATH)
print("Dataset Loaded ✔")

# ---------- Dataset Class ----------
class BrainDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------- Split ----------
train_idx, val_idx = train_test_split(range(len(X)), test_size=0.2)

train_data = BrainDataset(X[train_idx], y[train_idx])
val_data = BrainDataset(X[val_idx], y[val_idx])

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)

# ---------- Model ----------
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(16*63*63, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))

model = Model().to(device)

# ---------- Training ----------
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    total_loss = 0

    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(Xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}")

# ---------- Evaluation ----------
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for Xb, yb in val_loader:
        Xb = Xb.to(device)
        outputs = model(Xb)

        preds = (outputs > 0.5).float().cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(yb.numpy())

# ---------- Metrics ----------
acc = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print("\n🎯 Accuracy:", acc)
print("\n📊 Confusion Matrix:\n", cm)

# ---------- Save Model ----------
torch.save(model.state_dict(), "model.pth")
print("\nModel Saved ✅")