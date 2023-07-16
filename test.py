import torch
import torch.nn as nn
import utils
import torchvision.models
from classnames import CLASS_NAMES
import pandas as pd

from torch.utils.data import DataLoader
from torchvision import transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score,  classification_report

def load_model():

    # Load the pretrained MobileNetV2 model
    PATH = "model.pth"

    model = torchvision.models.mobilenet_v2() #for torch 1.9.0

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=len(CLASS_NAMES), bias=True)).to(device='cuda:0')
    
    model = utils.load_model(model=model, model_save_path=PATH)

    return model

def load_data():

    # Setup hyperparameters
    BATCH_SIZE = 64
    NUM_WORKERS = 4

    test_dir = "Test_Folder"

    # Setup directories
    test_transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])       
        ])
    
    test_dataset = ImageFolder(root= test_dir, 
                             transform = test_transform)
    
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, 
                                shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    
    return test_dataloader

def main():

    y_pred = []
    y_true = []

    device = "cuda:0"

    csv_name = "model.csv"

    model = load_model()

    test_dataloader = load_data()

    model.eval()

    classes = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y')

    with torch.inference_mode():

        for batch, (X, y) in enumerate(test_dataloader):
          
            X, y = X.to(device), y.to(device)
          
            model.to(device)
          
            output = model(X) # Feed Network
        
          #_, y_pred_class = torch.max(output, 1) # Get class index with highest probability

            y_pred_class = torch.argmax(torch.softmax(output, dim=1), dim=1)

            y_true.extend(y.tolist())

            y_pred.extend(y_pred_class.tolist())

        accuracy = accuracy_score(y_true, y_pred)

        string_accuracy = str(accuracy)

        report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)

        df = pd.DataFrame(report)

        print("accuracy is: " + string_accuracy)

        print("Printing classification Report")
        print(df)

        print(f"[INFO] Saving Classification Report to : {csv_name}")
        df.to_csv(csv_name, index=False)

    return None

if __name__ == "__main__":
    main()
