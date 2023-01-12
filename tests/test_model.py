import sys

sys.path.append('./src/models')
import torch
from model_lightningModule import Net


def test_model():
    batch_size = 1
    # Define an input to the model
    x = torch.randn((batch_size, 1, 28, 28))
    # Define the model
    model = Net(n_features = 784, n_hidden = [100, 100, 100], n_classes = 10, lr = 0.0001)
    # Run the model
    output = model(x)
    # Check that output size is correct
    assert list(output.shape) == [batch_size, 10], 'Model output shape should be [batch_size, 10]'


