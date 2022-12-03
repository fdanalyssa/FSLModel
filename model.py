import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import RMSprop
from torch.optim.lr_scheduler import StepLR
import torchvision
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from classnames import CLASS_NAMES
from torchinfo import summary

#Auto-Transform
weights = MobileNet_V2_Weights.DEFAULT

auto_transforms = MobileNet_V2_Weights.IMAGENET1K_V2.transforms

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
