import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pytest
import cv2

from data import BrainDataset, load_data, transform, IMG_SIZE


class TestBrainDataset:
    """Tests for the BrainDataset class."""

    def setup_method(self):
        self.X = torch.randn(10, 3, IMG_SIZE, IMG_SIZE)
        self.y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        self.dataset = BrainDataset(self.X, self.y)

    def test_len(self):
        """Dataset length should match number of samples."""
        assert len(self.dataset) == 10

    def test_getitem_returns_tuple(self):
        """__getitem__ should return (image, label) tuple."""
        item = self.dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 2

    def test_getitem_image_shape(self):
        """Retrieved image should have correct shape."""
        img, _ = self.dataset[0]
        assert img.shape == (3, IMG_SIZE, IMG_SIZE)

    def test_getitem_label_value(self):
        """Labels should match what was passed in."""
        for i in range(len(self.dataset)):
            _, label = self.dataset[i]
            assert label == self.y[i]

    def test_empty_dataset(self):
        """Empty dataset should have length 0."""
        empty_X = torch.empty(0, 3, IMG_SIZE, IMG_SIZE)
        empty_y = torch.tensor([])
        ds = BrainDataset(empty_X, empty_y)
        assert len(ds) == 0

    def test_single_item_dataset(self):
        """Single item dataset should work correctly."""
        single_X = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        single_y = torch.tensor([1])
        ds = BrainDataset(single_X, single_y)
        assert len(ds) == 1
        img, label = ds[0]
        assert img.shape == (3, IMG_SIZE, IMG_SIZE)
        assert label == 1

    def test_image_tensor_type(self):
        """Images should be float tensors."""
        img, _ = self.dataset[0]
        assert img.dtype == torch.float32

    def test_label_tensor_type(self):
        """Labels should be long/int tensors."""
        _, label = self.dataset[0]
        assert label.dtype in (torch.int64, torch.int32, torch.long)


class TestLoadData:
    """Tests for the load_data function."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.no_dir = os.path.join(self.temp_dir, "no")
        self.yes_dir = os.path.join(self.temp_dir, "yes")
        os.makedirs(self.no_dir)
        os.makedirs(self.yes_dir)

    def _create_test_image(self, path, size=(100, 100)):
        """Helper to create a test image file."""
        img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        cv2.imwrite(path, img)

    def test_load_basic(self):
        """Should load images and return tensors."""
        self._create_test_image(os.path.join(self.no_dir, "img1.jpg"))
        self._create_test_image(os.path.join(self.yes_dir, "img2.jpg"))

        X, y = load_data(self.temp_dir)

        assert isinstance(X, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert len(X) == 2
        assert len(y) == 2

    def test_load_correct_labels(self):
        """'no' folder should be label 0, 'yes' folder should be label 1."""
        self._create_test_image(os.path.join(self.no_dir, "healthy1.jpg"))
        self._create_test_image(os.path.join(self.no_dir, "healthy2.jpg"))
        self._create_test_image(os.path.join(self.yes_dir, "tumor1.jpg"))

        X, y = load_data(self.temp_dir)

        # First two images are from 'no' (label 0), last from 'yes' (label 1)
        assert y[0].item() == 0
        assert y[1].item() == 0
        assert y[2].item() == 1

    def test_load_image_shape(self):
        """Loaded images should be (3, IMG_SIZE, IMG_SIZE)."""
        self._create_test_image(os.path.join(self.no_dir, "img.jpg"))

        X, y = load_data(self.temp_dir)

        assert X.shape == (1, 3, IMG_SIZE, IMG_SIZE)

    def test_load_skips_invalid_files(self):
        """Should skip files that can't be read as images."""
        self._create_test_image(os.path.join(self.no_dir, "valid.jpg"))
        # Create an invalid file
        with open(os.path.join(self.no_dir, "invalid.txt"), "w") as f:
            f.write("not an image")

        X, y = load_data(self.temp_dir)

        assert len(X) == 1

    def test_load_multiple_images_per_folder(self):
        """Should load all valid images from each folder."""
        for i in range(5):
            self._create_test_image(os.path.join(self.no_dir, f"no_{i}.jpg"))
        for i in range(3):
            self._create_test_image(os.path.join(self.yes_dir, f"yes_{i}.jpg"))

        X, y = load_data(self.temp_dir)

        assert len(X) == 8
        assert (y == 0).sum().item() == 5
        assert (y == 1).sum().item() == 3

    def test_load_different_image_sizes(self):
        """Should handle images of different sizes (all resized to IMG_SIZE)."""
        self._create_test_image(os.path.join(self.no_dir, "small.jpg"), size=(50, 50))
        self._create_test_image(os.path.join(self.yes_dir, "large.jpg"), size=(300, 300))

        X, y = load_data(self.temp_dir)

        assert X.shape[2] == IMG_SIZE
        assert X.shape[3] == IMG_SIZE


class TestTransform:
    """Tests for the image transform pipeline."""

    def test_transform_output_shape(self):
        """Transform should produce (3, IMG_SIZE, IMG_SIZE) tensor."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = transform(img)
        assert result.shape == (3, IMG_SIZE, IMG_SIZE)

    def test_transform_output_range(self):
        """ToTensor should normalize to [0, 1]."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = transform(img)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_transform_output_type(self):
        """Transform output should be a float tensor."""
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = transform(img)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
