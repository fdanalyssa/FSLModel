"""
Trains a PyTorch image classification model using device-agnostic code.

"""

import os
import torch
import torch.nn as nn
from torch.optim import RMSprop
from torch.optim.lr_scheduler import StepLR
import data_setup, engine, model, utils

from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from classnames import CLASS_NAMES


# Setup hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_WORKERS = os.cpu_count()

# Setup directories
train_dir = "FSL_alphabet_and_nums/train"
val_dir = "FSL_alphabet_and_nums/test"


# Setup target device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Create transforms
train_transform, val_transform = data_setup.data_transform()

# Create datasets
train_dataset, val_dataset = data_setup.dataset_setup(train_transform, val_transform, train_dir, val_dir)



# Create DataLoaders with help from data_setup.py
train_dataloader, val_dataloader, class_names = data_setup.create_dataloader(train_dataset=train_dataset,
                                                val_dataset=val_dataset,
                                                batch_size=BATCH_SIZE,
                                                num_workers=NUM_WORKERS,
                                                pin_memory=True)

#Auto-Transform
weights = MobileNet_V2_Weights.DEFAULT

#Activates pretrained weights of model
model = mobilenet_v2(weights=weights).to(device='cuda:0')

#Freeze all layers
for param in model.features.parameters():
    param.requires_grad = False

#Adjusting the output layer  or changing the classifier portion
#set manual seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)

#Getting the length of classnames
output_size = len(CLASS_NAMES)

#Recreate the classifier layer and see it to the target device

model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=output_size, bias=True)).to(device='cuda:0')


# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)


# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=valdataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="saved_models",
                 model_name="FILOSign.pth")