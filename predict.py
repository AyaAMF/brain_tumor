import torch
from PIL import Image
from torchvision import transforms

from model import BrainTumorModel, IMG_SIZE

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])


def load_model(model_path, device=None):
    """Load a trained BrainTumorModel from disk."""
    if device is None:
        device = torch.device("cpu")
    model = BrainTumorModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_image(model, path):
    """Run prediction on a single image.

    Returns a tuple of (result_text, color) where color is 'red' for
    tumor detected and 'green' for healthy.
    """
    img = Image.open(path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)

    confidence = output.item()

    if confidence > 0.5:
        return f"Tumor Detected - Confidence: {confidence*100:.2f}%", "red"
    else:
        return f"Healthy Brain - Confidence: {(1-confidence)*100:.2f}%", "green"
