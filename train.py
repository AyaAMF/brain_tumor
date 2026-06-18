import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from model import BrainTumorModel, get_device
from data import BrainDataset, load_data


DATA_PATH = "brain_tumor_dataset"

device = get_device()

# ---------- Load Dataset ----------
print("Loading dataset...")
X, y = load_data(DATA_PATH)
print("Dataset Loaded")

# ---------- Split ----------
train_idx, val_idx = train_test_split(range(len(X)), test_size=0.2)

train_data = BrainDataset(X[train_idx], y[train_idx])
val_data = BrainDataset(X[val_idx], y[val_idx])

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)

# ---------- Model ----------
model = BrainTumorModel().to(device)

# ---------- Training ----------
criterion = torch.nn.BCELoss()
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

print(f"\nAccuracy: {acc}")
print(f"\nConfusion Matrix:\n{cm}")

# ---------- Save Model ----------
torch.save(model.state_dict(), "model.pth")
print("\nModel Saved")
