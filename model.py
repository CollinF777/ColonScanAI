# This is our tools for building neural networks
import torch.nn as nn
# This gives us access to existing models like ResNet18
from torchvision import models

from config import DEVICE

'''
This is we define our model
We are using ResNet18 with transfer learning
Transfer learning just means we are taking a model thats already been trained on one large dataset
and reusing it for a different task, instead of starting from scratch

We start from weights that are pretrained on ImageNet
freeze the backbone (every layer but the final) and replace the final layer to output our 2 classes
This makes it take much less data and training time compared to training from scratch
'''
def build_model(num_classes=2, freeze_backbone=True):
    # This is how we load ResNet18, the convolutional network designed for image recognition
    # The weights = part is just defining the set of weights we are starting with
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # This is how we freeze all the layers but the final
    if freeze_backbone:
        for param in model.parameters():
            # So for each of the models learnable parameters such as the weights and biases,
            # dont calculate the gradient for them so the optimizer wont update them during training
            param.requires_grad = False

    '''
    I'll just explain the next two lines in one big comment

    for num_features =
    model.fc is the models final layer
    .in_features tells us how many inputs that layer recieves, in ResNet18 thats 512
    So basically we're just saying how many features the model gives us to make its final decision
    This means before the model makes its final decision, it extracts 512 pieces of information about what it sees (features)
    In the end all this is doing is saying how many pieces of information the final layer recieves before it decides what
    class the image belongs to
    
    for model.fc =
    nn.Linear() creates a new layer
    num_features is the number of inputs aka 512 from the last line
    num_classes is how many categories we want (2)
    So we're replacing ResNet18's final layer which normally predicts 1000 ImageNet categories,
    with one that predicts our 2 categories
    '''
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    # This moves the model to either the cpu or gpu, ready to be user for training or predicting
    return model.to(DEVICE)