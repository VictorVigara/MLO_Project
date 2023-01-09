# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from torch.utils.data import Dataset


@click.group()
def cli():
    pass


@click.group()
def cli():
    pass

@click.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.argument('tensors_path', type=click.Path())
def main(data_path, tensors_path):
    print('Transforming raw data into tensors')
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')
    # Load data from data path
    data_train, data_test = [], []

    # Load test and train data independently 
    for file in os.listdir(data_path):
        if ('train' in file) == True:
            data_train.append(np.load(data_path+"/"+file))
        if ('test' in file) == True: 
            data_test.append(np.load(data_path+"/"+file))

    # Get train and test data as tensors
    images_train = torch.tensor(np.concatenate([d['images'] for d in data_train]), dtype=torch.float32).reshape(-1, 1, 28, 28)
    labels_train = torch.tensor(np.concatenate([d['labels'] for d in data_train]))
    
    images_test = torch.tensor(np.concatenate([d['images'] for d in data_test]), dtype=torch.float32).reshape(-1, 1, 28, 28)
    labels_test = torch.tensor(np.concatenate([d['labels'] for d in data_test]))

    
    # Save data as tensors
    torch.save(images_train, tensors_path + '/train_images.pt')
    torch.save(labels_train, tensors_path + '/train_labels.pt')
    torch.save(images_test, tensors_path + '/test_images.pt')
    torch.save(labels_test, tensors_path + '/test_labels.pt')

cli.add_command(main)

if __name__ == '__main__':
    cli()
    main()