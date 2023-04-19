import torch
import torch.quantization
import torchvision.models
import torch.nn as nn
from utils import load_model as load_model
from classnames import CLASS_NAMES

def quantize_model(model, PATH, input_shape, input_dtype):

    #load model
    model = load_model(model, PATH)

    model = model.to(device='cuda:0')

    #set model to eval mode
    model.eval()

    # Specify the input shape and data type
    input_shape = input_shape
    input_dtype = input_dtype

    # Prepare the model for quantization
    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')

    # 'fbgemm' for server, 'qnnpack' for mobile
    torch.backends.quantized.engine = 'qnnpack'

    quantized_model = torch.quantization.prepare(model, inplace=True)

    # Convert the model to a quantized format
    quantized_model = torch.quantization.convert(quantized_model, inplace=True)

    # Test the quantized model
    test_input = torch.randn(input_shape, dtype=input_dtype)
    quantized_model(test_input)

    target_dir = 'saved_models_quantized'

    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                            exist_ok=True)

    # Save the quantized model
    torch.save(quantized_model.state_dict(), 'saved_models_quantized/quantized_mobilenetv2_sample.pth')


def main():
    # Load the pretrained MobileNetV2 model
    PATH = "saved_models\FILOSign_RMS.pth"

    model = torchvision.models.get_model('mobilenet_v2')

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=len(CLASS_NAMES), bias=True)).to(device='cuda:0')

    input_shape = (1, 3, 224, 224)
    input_dtype = torch.float32

    quantize_model(model, PATH, input_shape=input_shape, input_dtype=input_dtype)


    

