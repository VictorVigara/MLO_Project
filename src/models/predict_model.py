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
@click.argument("model_checkpoint")
@click.option("--bs", default=32, help='batch size')

def predict(model_checkpoint, bs):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)
    curr_dir = os.getcwd()

    # Path to data
    data_path = "data/processed/"
    checkpoint_path = "src/models/train_models/"

    # Load model
    model = Net(784, [256, 132, 100], 10)

    # Load parameters to model
    checkpoint_dir = checkpoint_path+model_checkpoint
    state_dict = torch.load(checkpoint_dir)
    model.load_state_dict(state_dict)
    
    # Load test data
    test_images = torch.load('data/processed/test_images.pt')
    test_labels = torch.load('data/processed/test_labels.pt')

    transform = transforms.Normalize((0,), (1,))
    test_set = MyDataset(test_images, test_labels, transform)
    testloader = DataLoader(test_set, batch_size=bs, shuffle=True)

    images, labels = next(iter(testloader))
    # Get the class probabilities
    ps = torch.exp(model(images))
    # Most likely classes 
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    # Get model accuracy
    accuracy = torch.mean(equals.type(torch.FloatTensor))
    print(f'Accuracy: {accuracy.item()*100}%')


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

cli.add_command(predict)

if __name__ == '__main__': 
    cli()
    predict()