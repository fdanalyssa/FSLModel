#importing libraries and dependencies
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.io import read_image

#Creating a custom dataset class
class FSLData(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file) #reading the csv file
        self.img_dir = img_dir #path to the images
        self.transform = transform #transforming the images
        self.target_transform = target_transform #transforming the labels
        

    def __len__(self):
        #length of the dataset
        return len(self.img_labels) 
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) #getting the path of the image
        image = read_image(img_path) #reading the image
        label = self.img_labels.iloc[idx, 1] #getting the label of the image

        if self.transform:
            image = self.transform(image) #transforming the image
        if self.target_transform:
            label = self.target_transform(label) #transforming the label

        return image, label

#Augmenting the dataset
def transform():
    FSL_transforms_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                            std = [0.229, 0.224, 0.225])
        ])

    FSL_transforms_val = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return FSL_transforms_train, FSL_transforms_val





