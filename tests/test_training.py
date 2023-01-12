import sys

sys.path.append('./src/models')
import os.path

import pytest
import pytorch_lightning as pl
import torch
from model_lightningModule import Net
from myDataset import MyDataset
from torch.utils.data import DataLoader
from torchvision import transforms


@pytest.mark.skipif(not os.path.exists("./data"), reason="Data files not found")
def test_training():
     # Define normalization
    transform = transforms.Normalize((0,), (1,))
    # Create training dataset
    train_dataset = MyDataset('train', 'data/processed', transform)
    # Create training data loader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Create validation dataset
    val_dataset = MyDataset('test', 'data/processed', transform)
    # Create validation data loader
    val_loader = DataLoader(val_dataset, batch_size=32)
 
    # Create trainer
    trainer = pl.Trainer(max_epochs=1, limit_train_batches=0.1)
    # Define model
    model = Net(784, [100, 100, 100], 10, 0.0001)
    # Train model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    train_loss = trainer.logged_metrics['train_loss']
    val_loss = trainer.logged_metrics['val_loss']
    val_acc = trainer.logged_metrics['val_acc']

    assert train_loss >= 0, 'Training loss should be >= 0'
    assert val_loss >= 0, 'Validation loss should be >= 0'
    assert val_acc <= 1, 'Validation accuracy should be <= 1'
