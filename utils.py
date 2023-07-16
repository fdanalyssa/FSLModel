
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from torchinfo import summary
import pprint
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, classification_report
from torchvision import transforms
from PIL import Image


def save_model(model, target_dir, model_name):

  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)
  
 #loads pretrained model 
def load_model(model, model_save_path):
    
    print(f"[INFO] Loading model from: {model_save_path}")
    model.load_state_dict(torch.load(f=model_save_path))

    return model

def check_model(model, input_size, device):
    
      print("[INFO] Checking model")
      summary(model, input_size=input_size, col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], device=device)

      return None

def print_model_state_dict(model):
        
        print("[INFO] Printing model state dict")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(model.state_dict())

        return None


#Evaluation Metrics Functions

def confusion_matrix_plot(model, val_dataloader, device, cf_plot_name):

  y_pred = []
  y_true = []

  print("[INFO] Starting confusion matrix")

  # iterate over test data
  for batch, (X, y) in enumerate(val_dataloader):
          
          X, y = X.to(device), y.to(device)
          
          model.to(device)
          
          output = model(X) # Feed Network
        
          #_, y_pred_class = torch.max(output, 1) # Get class index with highest probability

          y_pred_class = torch.argmax(torch.softmax(output, dim=1), dim=1)

          y_true.extend(y.tolist())

          y_pred.extend(y_pred_class.tolist())

  # constant for classes
  classes = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y')

  # Build confusion matrix
  cf_matrix = confusion_matrix(y_true, y_pred)

  df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                      columns = [i for i in classes])
  
  plt.figure(figsize = (20,15))
  sn.heatmap(df_cm, annot=True)
  print("[INFO] Saving confusion matrix")
  plt.savefig(cf_plot_name)
    
  return None  

def Evaluation_Metrics(model, dataloader, device):

    y_pred = []
    y_true = []

    model.eval()

    # Turn on inference context manager
    with torch.inference_mode():
      # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):      

            X, y = X.to(device), y.to(device)

            outputs= model(X)

            y_pred_class = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

            y_true.extend(y.tolist())

            y_pred.extend(y_pred_class.tolist())


        classes = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y')
        
        f1 = f1_score(y_true, y_pred, average='macro')
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_macro = recall_score(y_true, y_pred, average='macro')

        report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)

        df = pd.DataFrame(report)

        return f1, precision_macro, recall_macro, df

def predict(model, image_path, classnames, device):
     
     img = Image.open(image_path)

     model.to(device)

     model.eval()

     data_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])
     
     with torch.inference_mode():
           
          transformed_image = data_transforms(img).to(device).float()

          transformed_image = transformed_image.unsqueeze(dim = 0)

          output = model(transformed_image).to(device)

          y_pred_class = torch.argmax(torch.softmax(output, dim=1), dim=1)

          prob = y_pred_class.max()

          y_pred_class = y_pred_class.tolist()

          print( f"the image is: {classnames[y_pred_class[0]]}")

     plt.figure()
     plt.imshow(img)
     plt.title(f"Pred: {classnames[y_pred_class[0]]} | Prob: {prob:.3f}")
     plt.axis(False)
     plt.show()
     
     


