import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch

'''
This is for importing the GradCAM class from pytorch_grad_cam
Its a class the generates a heatmap using information from the models gradients and internal
feature maps
In case you dont know what a heatmap is, its just highlighting regions that contributed
to a models prediction
'''
from pytorch_grad_cam import GradCAM
'''
This import brings in a helper function that overlays the heatmap onto the original image
This is much better than seeing the heatmap and image on their own 
'''
from pytorch_grad_cam.utils.image import show_cam_on_image
from config import DEVICE, IMAGE_SIZE, CLASSES
from data import eval_transform

def build_gradcam(model):
    target_layers = [model.layer4[-1]]
    return GradCAM(model = model, target_layers = target_layers)

def show_gradcam(model, cam, image_path, true_label_idx):
    model.eval()

    image = Image.open(image_path).convert('RGB')
    input_tensor = eval_transform(image).unsqueeze(0).to(DEVICE)
    # Since the backbone is frozen we need to specify this so we dont lose our gradients which could cause an error
    # this might not cause an error but I wanted to be safe
    input_tensor.requires_grad_(True)
    grayscale_cam = cam(input_tensor = input_tensor) [0, :]

    rgb_image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    rgb_image = np.array(rgb_image).astype(np.float32) / 255.0

    #overlays heatmap over the image
    overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

    #models prediction
    with torch.no_grad():
        scores = model(input_tensor)
    pred_idx = scores.argmax(dim = 1).item()
    confidence = torch.softmax(scores, dim = 1)[0, pred_idx].item()

    #shows both the original and Grad-CAM
    plt.figure(figsize = (10,5))

    plt.subplot(1, 2, 1)
    plt.imshow(rgb_image)
    # Show the true label so the plot can be compared against the prediction
    plt.title(f'Original Image (pred: {CLASSES[true_label_idx]}')

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    # SHow which class was predicted with confidence
    plt.title(f"Grad-CAM: (pred: {CLASSES[pred_idx]}, {confidence:.1%})")

    plt.tight_layout()
    plt.show()

def show_gradcam_samples(model, test_paths, test_labels, num_samples = 3):
    cam = build_gradcam(model)

    #picks random test image
    num_samples = min(num_samples, len(test_paths))
    sample_indices = random.sample(range(len(test_paths)), num_samples)

    #shows the grad cam for each test image
    for i in sample_indices:
        show_gradcam(model, cam, test_paths[i], test_labels[i])