import torch
import torch.nn
from torchvision.models import MobileNet_V2_Weights
import matplotlib.pyplot as plt
from PIL import Image

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 1. Take in a trained model, class names, image path, image size, a transform and target device
def pred_and_plot_image(model, image_path, class_names, image_size, device):
    
    
    # 2. Open image
    img = Image.open(image_path)

    # 3. Create transformation for image (if one doesn't exist)
  #  if transform is not None:
    #   image_transform = transform
  #  else:
    image_transform = MobileNet_V2_Weights.IMAGENET1K_V2.transforms

    ### Predict on image ### 

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    
    with torch.inference_mode():
      # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
     # transformed_image = image_transform(img).unsqueeze(dim=0)
      transformed_image = image_transform(img).float()
      transformed_image = transformed_image.unsqueeze(0)
      # 7. Make a prediction on image with an extra dimension and send it to the target device
      target_image_pred = model(transformed_image.to(device))

    # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 9. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 10. Plot image with predicted label and probability 
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
    plt.axis(False);
