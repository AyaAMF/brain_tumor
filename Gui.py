import torch
import torch.nn as nn
from PIL import Image, ImageTk
import customtkinter as ctk
from tkinter import filedialog
from torchvision import transforms

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
model = Model()
model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
model.eval()

# ---------------- IMAGE TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ---------------- PREDICTION ----------------
def predict_image(path):
    img = Image.open(path).convert("RGB")
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

    if file_path:
        # Display Image
        img = Image.open(file_path)
        img = img.resize((220, 220))
        photo = ImageTk.PhotoImage(img)

        image_label.configure(image=photo, text="")
        image_label.image = photo

        # Prediction
        result, color = predict_image(file_path)
        result_label.configure(text=result, text_color=color)

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