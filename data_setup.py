
#importing libraries and dependencies

from torch.utils.data import DataLoader
from torchvision import transforms as transforms
from torchvision.datasets import ImageFolder
from classnames import CLASS_NAMES

#Auto-Transform
#weights = models.MobileNet_V2_Weights.DEFAULT

#auto_transforms = weights.transforms()


#Augmenting the dataset
def data_transform():

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])
        ])

    val_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])       
        ])

    return train_transform, val_transform


#Uses ImageFolder to create the datasets
def dataset_setup(train_transform, val_transform, train_dir = "FSL_alphabet/train", val_dir = "FSL_alphabet/val"):

    #datasets
    train_dataset = ImageFolder(root=train_dir,
                                transform = train_transform)

    val_dataset = ImageFolder(root= val_dir, 
                             transform = val_transform)

    return train_dataset, val_dataset


#Turns the datasets into dataloaders
def FSLdataloader(train_dataset, val_dataset, batch_size=32, num_workers=4, pin_memory=True):

    class_names = CLASS_NAMES

    #dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, 
                                shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_dataloader, val_dataloader, class_names

 








