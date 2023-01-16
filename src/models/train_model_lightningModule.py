import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from model_lightningModule import Net
from myDataset import MyDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from torchvision import transforms


def train(): 
    # Define normalization ()
    transform = transforms.Normalize((0,), (1,))
    # Create training dataset
    train_dataset = MyDataset('train', 'data/processed', transform)
    # Create training data loader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Create validation dataset
    val_dataset = MyDataset('test', 'data/processed', transform)
    # Create validation data loader
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Define callback
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")

 
    # Create trainer
    trainer = pl.Trainer(max_epochs=2, accelerator="gpu", limit_train_batches=1.0, callbacks=[early_stopping_callback], default_root_dir="src/models/train_models/pytorch_lightning")
    # Define model
    model = Net(784, [100, 100, 100], 10, 0.0001)
    # Train model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    
    

train()