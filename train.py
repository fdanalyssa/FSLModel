"""
Trains a PyTorch image classification model using device-agnostic code.

"""
import torch
import torch.nn as nn
import torch.optim
import data_setup, engine

from torchvision.models import mobilenet_v2

from classnames import CLASS_NAMES

#from torchinfo import summary

def main():
    # Setup hyperparameters
    NUM_EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0001
    NUM_WORKERS = 4
    ARCHITECTURE = "CNN_MobileNetV2"
    DATASET = "FSL_Resized_400"
    OPTIMIZER_NAME = "RMS"

    # Setup directories
    train_dir = "FSL_alphabet_400/train"
    val_dir = "FSL_alphabet_400/val"

    # Setup target device
    device = "cuda:0"

    # Create transforms
    train_transform, val_transform = data_setup.data_transform()

    # Create datasets
    train_dataset, val_dataset = data_setup.dataset_setup(train_transform, val_transform, train_dir, val_dir)



    # Create DataLoaders with help from data_setup.py
    train_dataloader, val_dataloader, class_names = data_setup.FSLdataloader(train_dataset=train_dataset,
                                                    val_dataset=val_dataset,
                                                    batch_size=BATCH_SIZE,
                                                    num_workers=NUM_WORKERS,
                                                    pin_memory=True)

    #Auto-Transform
  #  weights = MobileNet_V2_Weights.IMAGENET1K_V1

 #   torch.backends.quantized.engine = 'fbgemm'
    
    #Quantized Weights
  #  quantized_weights = quantization.MobileNet_V2_QuantizedWeights.DEFAULT


    #Activates pretrained weights of model
  #  model = mobilenet_v2(weights=weights).to(device='cuda:0')
    model = mobilenet_v2(pretrained=True).to(device='cuda:0')
    

   # model = quantization.mobilenet_v2(weights=quantized_weights, quantize=True).to(device='cuda:0')

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

    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-5)
   # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # learning rate scheduler
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Start training with help from engine.py
    engine.train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=val_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs= NUM_EPOCHS,
                device=device,
                lr=LEARNING_RATE, 
                architecture=ARCHITECTURE,
                dataset=DATASET,
                opt_name=OPTIMIZER_NAME,
                model_path="saved_models/FILOSign_RMS_0.0001_0.2_20e_400px_androidtest.pth"
                )    


if __name__ == "__main__":

    main()