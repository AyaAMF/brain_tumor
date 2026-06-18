import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pytest
from PIL import Image

from model import BrainTumorModel, IMG_SIZE
from predict import predict_image, load_model, transform


class TestPredictImage:
    """Tests for the predict_image function."""

    def setup_method(self):
        self.model = BrainTumorModel()
        self.model.eval()
        # Create a temporary test image
        self.temp_dir = tempfile.mkdtemp()
        self.test_img_path = os.path.join(self.temp_dir, "test.jpg")
        img = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )
        img.save(self.test_img_path)

    def test_predict_returns_tuple(self):
        """predict_image should return a (text, color) tuple."""
        result = predict_image(self.model, self.test_img_path)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_predict_color_is_valid(self):
        """Color should be either 'red' or 'green'."""
        _, color = predict_image(self.model, self.test_img_path)
        assert color in ("red", "green")

    def test_predict_text_contains_confidence(self):
        """Result text should contain 'Confidence'."""
        text, _ = predict_image(self.model, self.test_img_path)
        assert "Confidence" in text

    def test_predict_tumor_detected_format(self):
        """When confidence > 0.5, should indicate tumor detected."""
        # Force high confidence by manipulating model weights
        with torch.no_grad():
            self.model.fc.bias.fill_(10.0)  # Push sigmoid output > 0.5

        text, color = predict_image(self.model, self.test_img_path)
        assert "Tumor Detected" in text
        assert color == "red"

    def test_predict_healthy_format(self):
        """When confidence <= 0.5, should indicate healthy."""
        # Force low confidence
        with torch.no_grad():
            self.model.fc.bias.fill_(-10.0)  # Push sigmoid output < 0.5

        text, color = predict_image(self.model, self.test_img_path)
        assert "Healthy Brain" in text
        assert color == "green"

    def test_predict_with_different_image_sizes(self):
        """Should work with images of various sizes."""
        for size in [(50, 50), (200, 200), (640, 480)]:
            img_path = os.path.join(self.temp_dir, f"test_{size[0]}x{size[1]}.jpg")
            img = Image.fromarray(
                np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            )
            img.save(img_path)
            result = predict_image(self.model, img_path)
            assert isinstance(result, tuple)

    def test_predict_with_png(self):
        """Should work with PNG images."""
        png_path = os.path.join(self.temp_dir, "test.png")
        img = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )
        img.save(png_path)
        result = predict_image(self.model, png_path)
        assert isinstance(result, tuple)

    def test_predict_confidence_percentage_format(self):
        """Confidence should be formatted as a percentage."""
        text, _ = predict_image(self.model, self.test_img_path)
        assert "%" in text


class TestLoadModel:
    """Tests for the load_model function."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.pth")
        # Save a model for testing
        model = BrainTumorModel()
        torch.save(model.state_dict(), self.model_path)

    def test_load_model_returns_model(self):
        """load_model should return a BrainTumorModel instance."""
        model = load_model(self.model_path)
        assert isinstance(model, BrainTumorModel)

    def test_load_model_eval_mode(self):
        """Loaded model should be in eval mode."""
        model = load_model(self.model_path)
        assert not model.training

    def test_load_model_with_device(self):
        """Should respect the device parameter."""
        device = torch.device("cpu")
        model = load_model(self.model_path, device=device)
        # Check all parameters are on CPU
        for param in model.parameters():
            assert param.device.type == "cpu"

    def test_load_model_produces_output(self):
        """Loaded model should produce valid output."""
        model = load_model(self.model_path)
        x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        output = model(x)
        assert output.shape == (1, 1)
        assert 0 <= output.item() <= 1


class TestPredictTransform:
    """Tests for the prediction transform pipeline."""

    def test_transform_pil_image(self):
        """Transform should work with PIL images."""
        img = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )
        result = transform(img)
        assert result.shape == (3, IMG_SIZE, IMG_SIZE)

    def test_transform_preserves_channels(self):
        """Transform should preserve RGB channels."""
        img = Image.fromarray(
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        )
        result = transform(img)
        assert result.shape[0] == 3
