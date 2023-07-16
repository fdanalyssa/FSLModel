import torch
import torch.nn as nn
import torchvision.models 
from torch.utils.mobile_optimizer import optimize_for_mobile
from classnames import CLASS_NAMES
from utils import load_model as load_model
from utils import check_model as check_model

def script_model(model):

    # Define an example input
    input_shape = (1, 3, 224, 224)
    input_data = torch.randn(input_shape)

    model = model.to(device='cpu')
    
    model.eval()

    # Convert the model to TorchScript
    traced_model = torch.jit.trace(model, input_data)

    # Save the TorchScript model to a file
    traced_model.save('model.pt')

    traced_script_module_optimized = optimize_for_mobile(traced_model)

    traced_script_module_optimized._save_for_lite_interpreter("model.ptl")
    

def main():

    # Load the pretrained MobileNetV2 model
    PATH = "model.pth"


    model = torchvision.models.mobilenet_v2() 

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=len(CLASS_NAMES), bias=True)).to(device='cuda:0')
    
    model = load_model(model=model, model_save_path=PATH)

    script_model(model)

    return None


if __name__ == "__main__":

    main()

