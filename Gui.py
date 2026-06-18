import sys
import logging
import torch
import torch.nn as nn
from PIL import Image, ImageTk
import customtkinter as ctk
from tkinter import filedialog, messagebox
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------- SETTINGS ----------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

device = torch.device("cpu")
IMG_SIZE = 128

# ---------------- MODEL ----------------
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 63 * 63, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))

# ---------------- LOAD MODEL ----------------
MODEL_PATH = "model.pth"
try:
    model = Model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
except FileNotFoundError:
    logger.error("Model file not found: %s. Run train.py first.", MODEL_PATH)
    sys.exit(1)
except (RuntimeError, Exception) as exc:
    logger.error("Failed to load model from %s: %s", MODEL_PATH, exc)
    sys.exit(1)

# ---------------- IMAGE TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ---------------- PREDICTION ----------------
def predict_image(path):
    try:
        img = Image.open(path).convert("RGB")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot open image: {exc}") from exc

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)

    confidence = output.item()

    if confidence > 0.5:
        return f"🚨 Tumor Detected\nConfidence: {confidence*100:.2f}%", "red"
    else:
        return f"✅ Healthy Brain\nConfidence: {(1-confidence)*100:.2f}%", "green"

# ---------------- UPLOAD FUNCTION ----------------
def upload_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )

    if not file_path:
        return

    try:
        img = Image.open(file_path)
        img = img.resize((220, 220))
        photo = ImageTk.PhotoImage(img)

        image_label.configure(image=photo, text="")
        image_label.image = photo

        result, color = predict_image(file_path)
        result_label.configure(text=result, text_color=color)
    except (ValueError, OSError) as exc:
        logger.error("Error processing image %s: %s", file_path, exc)
        messagebox.showerror(
            "Image Error",
            f"Could not process the selected image:\n{exc}"
        )
    except Exception as exc:
        logger.error("Unexpected error during prediction: %s", exc)
        messagebox.showerror(
            "Prediction Error",
            f"An unexpected error occurred:\n{exc}"
        )

# ---------------- APP ----------------
app = ctk.CTk()
app.title("Brain Tumor Detection AI")
app.geometry("600x700")
app.resizable(False, False)

# ---------------- TITLE ----------------
title = ctk.CTkLabel(
    app,
    text="🧠 Brain Tumor Detection System",
    font=("Arial", 28, "bold")
)
title.pack(pady=25)

# ---------------- IMAGE BOX ----------------
image_label = ctk.CTkLabel(
    app,
    text="MRI Image Preview",
    width=250,
    height=250,
    fg_color="#1e1e1e",
    corner_radius=15
)
image_label.pack(pady=20)

# ---------------- BUTTON ----------------
upload_btn = ctk.CTkButton(
    app,
    text="📂 Upload MRI Image",
    command=upload_image,
    width=250,
    height=50,
    font=("Arial", 18, "bold"),
    corner_radius=15
)
upload_btn.pack(pady=25)

# ---------------- RESULT ----------------
result_label = ctk.CTkLabel(
    app,
    text="",
    font=("Arial", 22, "bold")
)
result_label.pack(pady=30)

# ---------------- FOOTER ----------------
footer = ctk.CTkLabel(
    app,
    text="AI Medical Assistant",
    font=("Arial", 14),
    text_color="gray"
)
footer.pack(side="bottom", pady=15)

# ---------------- RUN ----------------
app.mainloop()