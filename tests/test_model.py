import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from model import BrainTumorModel, IMG_SIZE


class TestBrainTumorModel:
    """Tests for the BrainTumorModel CNN."""

    def setup_method(self):
        self.model = BrainTumorModel()
        self.model.eval()

    def test_model_instantiation(self):
        """Model should initialize without errors."""
        model = BrainTumorModel()
        assert model is not None

    def test_forward_pass_single_image(self):
        """Forward pass with a single image should return shape (1, 1)."""
        x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        output = self.model(x)
        assert output.shape == (1, 1)

    def test_forward_pass_batch(self):
        """Forward pass with a batch should return shape (batch_size, 1)."""
        batch_size = 8
        x = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE)
        output = self.model(x)
        assert output.shape == (batch_size, 1)

    def test_output_range_sigmoid(self):
        """Output should be in [0, 1] due to sigmoid activation."""
        x = torch.randn(16, 3, IMG_SIZE, IMG_SIZE)
        output = self.model(x)
        assert (output >= 0).all()
        assert (output <= 1).all()

    def test_output_dtype(self):
        """Output should be float tensor."""
        x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        output = self.model(x)
        assert output.dtype == torch.float32

    def test_model_parameters_exist(self):
        """Model should have trainable parameters."""
        params = list(self.model.parameters())
        assert len(params) > 0
        total_params = sum(p.numel() for p in params)
        assert total_params > 0

    def test_conv_layer_config(self):
        """Conv layer should have correct in/out channels and kernel size."""
        assert self.model.conv.in_channels == 3
        assert self.model.conv.out_channels == 16
        assert self.model.conv.kernel_size == (3, 3)

    def test_pool_layer_config(self):
        """MaxPool layer should have kernel_size 2."""
        assert self.model.pool.kernel_size == 2
        assert self.model.pool.stride == 2

    def test_fc_layer_config(self):
        """Fully connected layer should map to 1 output."""
        assert self.model.fc.out_features == 1
        assert self.model.fc.in_features == 16 * 63 * 63

    def test_gradient_flow(self):
        """Gradients should flow through all parameters during backprop."""
        self.model.train()
        x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
        target = torch.tensor([[1.0], [0.0]])

        output = self.model(x)
        loss = torch.nn.BCELoss()(output, target)
        loss.backward()

        for param in self.model.parameters():
            assert param.grad is not None
            assert param.grad.abs().sum() > 0

    def test_model_deterministic_eval(self):
        """Model in eval mode should produce same output for same input."""
        x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        out1 = self.model(x)
        out2 = self.model(x)
        assert torch.allclose(out1, out2)

    def test_model_state_dict(self):
        """Model state dict should contain expected keys."""
        state = self.model.state_dict()
        expected_keys = {'conv.weight', 'conv.bias', 'fc.weight', 'fc.bias'}
        assert expected_keys == set(state.keys())
