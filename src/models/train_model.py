import argparse
import os
import sys

import click
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from model import Net
from sklearn import metrics
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
@click.option("--ep", default=30, help='epochs for training')
@click.option("--bs", default=32, help='batch size')
def train(lr, bs, ep):
    ''' Train function load tensors from data/processed and train a network 
        defined in models.py.  At the end some figures are generated with the results'''

    print("Starting training")
    print(f"Learning rate: {lr}  Batch size: {bs}  Epochs: {ep}")
    curr_dir = os.getcwd()

    # Path to data
    data_path = "data/processed/"

    # Define model
    model = Net(784, [256, 132, 100], 10)
    # Load train and test data
    train_imgs = torch.load(data_path+'train_images.pt')
    train_labels = torch.load(data_path+'train_labels.pt')

    test_imgs = torch.load(data_path+'test_images.pt')
    test_labels = torch.load(data_path+'test_labels.pt')

    # Define normalization
    transform = transforms.Normalize((0,), (1,))

    # Load train dataset
    train_set = MyDataset(train_imgs, train_labels, transform)
    trainloader = DataLoader(train_set, batch_size=bs, shuffle=True)

    # Load test data
    test_set = MyDataset(test_imgs, test_labels, transform)
    testloader = DataLoader(test_set, batch_size=bs, shuffle=True)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)


    best_test_loss = 0
    train_loss, test_loss = [], []
    train_acc, test_acc = [], []
    for e in range(ep):

        # TRAINING
        train_curr_loss = 0
        train_preds, train_targs = [], []

        model.train()
        for images, label in trainloader:
            # Reshape image to enter FFNN
            images = images.view(images.shape[0], -1)
            # Set gradients to zero
            optimizer.zero_grad()
            # Predict 
            output = model(images)
            t_preds = torch.max(output,1)[1]
            # Calculate loss
            loss = criterion(output, label)
            # Calculate gradients
            loss.backward()
            # Update weights
            optimizer.step()
            # Get loss from all batch
            train_curr_loss += loss.item()

            train_targs += list(label.numpy())
            train_preds += list(t_preds.numpy())
        # Compute train batch loss and accuracy
        train_batch_loss = train_curr_loss/len(trainloader)
        train_loss.append(train_batch_loss)

        train_batch_acc = metrics.accuracy_score(train_targs, train_preds)
        train_acc.append(train_batch_acc)

        # VALIDATION
        test_curr_loss = 0
        test_preds, test_targs = [], []
        model.eval()
        for images, labels in testloader:
            images = images.view(images.shape[0], -1)
            # Predict 
            output = model(images)
            te_preds = torch.max(output,1)[1]
            # Calculate loss
            loss = criterion(output, labels)

            test_curr_loss += loss.item()

            test_targs += list(labels.numpy())
            test_preds += list(te_preds.numpy())

        # Compute test batch loss and accuracy
        test_batch_loss = test_curr_loss/(len(testloader))    
        test_loss.append(test_batch_loss)

        test_batch_acc = metrics.accuracy_score(test_targs, test_preds)
        test_acc.append(test_batch_acc)
        
        print(f"Epoch {e+1} Train Loss =  {train_batch_loss}  Train Acc = {train_batch_acc} Test Loss  =  {test_batch_loss}  Test Acc  = {test_batch_acc}")
        print("--------------------------------------------------------------------------")

        # Save best model (lowest test loss)
        if test_batch_loss > best_test_loss: 
            best_test_loss = test_batch_loss
            save_dir = "src/models/train_models/best_checkpoint.pth"
            torch.save(model.state_dict(), save_dir)
    
    plt.figure()
    plt.plot(range(ep), train_loss, 'b', test_loss, 'r')
    plt.legend(['Train Loss','Test Loss'])
    plt.xlabel('Updates'), plt.ylabel('Loss')
    plt.title('Loss')
    plt.savefig('reports/figures/loss.png')

    plt.figure()
    plt.plot(range(ep), train_acc, 'b', test_acc, 'r')
    plt.legend(['Train Accuracy','Test Accuracy'])
    plt.xlabel('Updates'), plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.savefig('reports/figures/accuracy.png')

class MyDataset(Dataset): 
    def __init__(self, images, labels, transform):
        super().__init__()
        self.images, self.labels = images, labels
        self.transform = transform

    def __len__(self): 
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform is not None: 
            image = self.transform(image)

        return image, label

cli.add_command(train)

if __name__ == '__main__': 
    cli()
    train()