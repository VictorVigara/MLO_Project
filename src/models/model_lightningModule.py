import torch.nn.functional as F
from torch import nn
from torch import optim
import pytorch_lightning as pl
import torch


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
        # Check dimensions
        if x.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError('Expected each sample to have shape [1, 28, 28]')

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
        self.log("train_loss", float(loss))

        return loss

    def configure_optimizers(self): 
        return optim.Adam(self.parameters(), lr=self.lr)

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        preds = self(images)
        loss = self.criterion(preds, labels)
        # Calculate predictions probabilities
        ps = torch.exp(preds)
        # Most likely classes 
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        # Get model accuracy
        accuracy = torch.mean(equals.type(torch.FloatTensor))

        # Log metrics
        self.log("val_loss", float(loss))
        self.log("val_acc", float(accuracy))