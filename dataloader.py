#importing libraries and dependencies
import os
import pandas as pd

import torch
from torch.utils.data import DataLoader as dataloaders

import torchvision
from torchvision import transforms as transforms
from torchvision.io import read_image
from torchvision.datasets import ImageFolder

#Augmenting the dataset

def data_transform():
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])
        ])

    val_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    return train_transform, val_transform

def dataset_setup():

    train_transform, val_transform = data_transform()

    #datasets
    train_dataset = torchvision.datasets.ImageFolder(root='FSL_alphabet_and_nums/train',
                                                    transform = train_transform)

    val_dataset = torchvision.datasets.ImageFolder(root='FSL_alphabet_and_nums/val', 
                                                    transform = val_transform)
    return train_dataset, val_dataset


def dataloader(bata_size=64):

    train_dataset, val_dataset = dataset_setup()

    #dataloader
    train_dataloader = dataloaders(dataset=train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=4, pin_memory=True)

    val_dataloader = dataloaders(dataset=val_dataset, batch_size=batch_size, 
                                shuffle=False, num_workers=4, pin_memory=True)

    return train_dataloader, val_dataloader

 








