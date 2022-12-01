#importing libraries and dependencies
import os
import time
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler

import torchvision
import torchvision.models as models
from torchvision import datasets, transforms

from dataloader import FSLData, transform

# Activates gpu if available
device = "cuda:0" if torch.cuda.is_available() else "cpu"

#Dataloader
FSL_dataset = Dataset.FSLData(annotations_file='FSLdata.csv', img_dir="asl_alphabet/train", transform = FSL_transforms_train)

dataset_loader = torch.utils.data.DataLoader(dataset = FSL_dataset, batch_size=4, shuffle=True, num_workers=4)

#Log in to wandb
wandb.login()

# Setting up the mobilenet_v2 model
model = models.mobilenet_v2(pretrained=True)

def train_model():
    pass
    return None

def val_model():
    pass
    return None