'''
Yousif this is your other part
This should be done second as it needs Camerons part to be done as you need the eval_transform from it

This file is for the Grad-CAM explainability
It is meant to visualize what parts of an image most influenced the models prediction as a heatmap overlay

If you dont know an import and its not commented, check a diff file cuz im just not recommenting the same thing
it might be in there somewhere
'''
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
# uncomment when cameron is done
# from data import eval_transform

'''
To start you need a simple build_gradcam function that has model as a param
all it needs to do is target the last convolutional block of ResNet 18
That sounds scary but to save you the effort its just [model.layer4[-1]]
Use that to return a GradCAM
'''

'''
The next function I would call something like show_gradcam
It would need model, cam, image_path, and true_label_idx as parameters
All the params are self-explanatory except for true_label_idx, thats just the index representing the images correct class

Inside the function you should open the image and convert it to RGB
You can then get the input tensor by calling camerons eval_transform on it and using unsqueeze(0) to say that the batch 
contains one image as PyTorch models are going to want a batch dimension even if its just one
then just use .to(DEVICE) to move the tensor to the device the model is using

After that you can make the heatmap and select the one for our image 
Then you're free to prepare the image, resize it to the correct image size, convert it to a numpy array, and convert the 
rgb from 0 to 255 to 0 to 1 because the show_cam_on_image() function is expecting the original image to be represented with 0-1 RGB
Make sure you overlay the heatmap after

Then just send the prepared image into the model (dont forget to turn off gradients) and get the scores from it
Also make sure you get the index of the class with the highest score

After that you can use plt to make a figure with two images
One would be the original image and the other would be the grad-cam image
'''

'''
I would add in one final function called something like show_gradcam_samples
It would have the params model, test_paths, test_labels, and num_samples (which would default to 3)
In case the params arent obvious, model is our trained model, test_paths is the paths to the images in the test dataset,
test_labels is the list of correct labels for those images, and num_samples would be how many images to display

You would have to call build_gradcam at the start of this so its set up
and then for a random sample go through and call show_gradcam for each image until you reach the number of samples
'''