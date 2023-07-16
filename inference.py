
import torch.nn as nn
import torchvision.models
from classnames import CLASS_NAMES
import utils
import torch

def main():
    # Load the pretrained MobileNetV2 model
    PATH = "model.pt"

    model = torchvision.models.mobilenet_v2() #for torch 1.9.0

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=len(CLASS_NAMES), bias=True)).to(device='cuda:0')
    
    #model = utils.load_model(model=model, model_save_path=PATH)

    #model = torch.jit.load(PATH)

    utils.predict(model=model,
                    image_path="Image.jpg",
                    classnames=CLASS_NAMES,
                    device="cuda:0")

if __name__ == "__main__":

    main()
