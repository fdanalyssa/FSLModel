import torch.nn as nn
import utils as utils
import inference as inference
from classnames import CLASS_NAMES
from torchvision.models import get_model



def main():

    PATH = "saved_models\FILOSign_RMS.pth"

    model = get_model('mobilenet_v2')

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=len(CLASS_NAMES), bias=True)).to(device='cuda:0')


    model = utils.load_model(model, PATH)
    image_path = "FSL_alphabet/val/A/A1683368276-IMG20230406174613_BURST000_COVER.jpg"
    class_names = CLASS_NAMES
    image_size = 224
    device = "cuda:0"

    inference.pred_and_plot_image(model, image_path, class_names, image_size, device)

    return None

main()