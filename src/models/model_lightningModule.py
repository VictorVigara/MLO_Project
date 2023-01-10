import torch.nn.functional as F
from torch import nn
from torch import optim
import pytorch_lightning as pl


class Net(pl.LightningModule):
    def __init__(self, n_features, n_hidden, n_classes, lr):
        super().__init__()

        self.fc1 = nn.Linear(n_features, n_hidden[0])
        self.fc2 = nn.Linear(n_hidden[0], n_hidden[1])
        self.fc3 = nn.Linear(n_hidden[1], n_hidden[2])
        self.fc4 = nn.Linear(n_hidden[2], n_classes)

        self.lr = lr
        self.dropout = nn.Dropout(p=0.3)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x

    def training_step(self, batch, batch_idx): 
        images, labels = batch
        preds = self(images)
        loss = self.criterion(preds, labels)
        return loss

    def configure_optimizers(self): 
        return optim.Adam(self.parameters(), lr=self.lr)

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        preds = self(images)
        loss = self.criterion(preds, labels)
        self.log("val_loss", loss)