# from test import _PATH_DATA
# print(_PATH_DATA)
import sys
sys.path.append("./src/models/")

import os
import torch
from myDataset import MyDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import pytest

@pytest.mark.skipif(not os.path.exists("./data"), reason="Data files not found")
def test_data():
    # Defend directories to load data
    data_dir = 'data/processed'

    transform = transforms.Normalize((0,), (1,))

    # Load train and test dataset
    train_dataset = MyDataset('train',data_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=1)

    test_dataset = MyDataset('test',data_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=1)


    for images, labels in train_loader:
        assert list(images.shape) == [1, 1, 28, 28], 'Images shape should be [batch_size, 1, 28, 28]'
        assert len(labels) == 1, 'Labels size should be batch_size'
    
    for images, labels in test_loader: 
        assert list(images.shape) == [1, 1, 28, 28], 'Images shape should be [batch_size, 1, 28, 28]'
        assert len(labels) == 1, 'Labels size should be batch_size'

    assert len(train_dataset) == 40000, 'Train dataset length should be 40000'
    assert len(test_dataset) == 5000, 'Test dataset length should be 5000'

    