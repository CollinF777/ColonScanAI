'''
This is the shared config for the entire project
Im going to comment everything very heavily so you guys dont ever get lost
We are using relative pathing so make sure you verify your file structure
with me first, although we should set it up in class --Collin
'''
import torch

'''
These are our paths
Each folder should contain colon_aca/ and colon_n/ subfolders
So you should have train and test folders then those below that
You can delete the lung folders but if you didnt theyll be ignored anywatds
'''
TRAINVAL_DIR = "./data/trainval" # Train and validation images, this should be split in data.py whoever is doing that
TEST_DIR = "./data/test" # Held out test images
'''
Dont worry too much about this model path
This is just going to hold the models learned weights
AKA the numbers inside the network that training adjusted, its a saved snapshot of what the model learned

Im not going to get too into it but its a neural network and those are just big piles of math in the end
images go in as numbers, get multiplied and added through many layers and then a prediction comes out
Those weights are the numbers being used in those multiplications
It determines how much each piece of information matters

For example, say we had:
score = (0.2 × gland_disorganization) + (0.7 × nucleus_size) + (0.1 × stain_darkness)
The weights would be 0.2, 0.7, and 0.1
It tells us that nucleus size matters the most
With our model its doing the same thing but with about 11 million weights stacked across many layers

Our training is going to adjust our weights in the final layer so we need to save it here
The weights start as pretrained values (taken from ImageNet) in the backbones plus random values in our new final layer
The model looks at the batches of images and makes at prediction. At first that final layer is going to make bad guesses
The loss function then measures how wrong the predictions were.
The optimizer will nudge the weights slightly to make those predictions just a bit less wrong
Then we repeat that thousands of times over 10 epochs and the weights will gradually settle into values that seperate cancerous and beign tissue
Thats the learning, the weight changing. Nothing else in the model will change

For the record, in our project we're using transfer learning which means we're taking a model thats already been trained on
one large dataset and reusing it for a different, smaller task instead of creating an entirely new model.
The idea is that a model trained on millions of everyday photos has already learned to detect general visual patterns like edges, textures, shapes, and color gradients
This ends up being useful for any image task 
We are starting with ResNet18 which is a model thats been trained to recognize patterns already in 1.2 million everyday images
We freeze most of the layers to keep everything its already learned
Then we replace the final layer with a new one the outputs our 2 classes (cancerous vs benign) instead of the original 1,000 ImageNet categories
So we train only the new final layer on on our colon histopathology images

I also just realized I didnt explain what ResNet18 and ImageNet is
ResNet18 is the model and ImageNet is the dataset of 1.2 million everyday photos
ImageNet is what ResNet18 is trained on

Also an epoch is one complete pass through the entire training set
Basically when the model has seen every training image once, thats one epoch

All this to say, once we're done and I make main.py and its finished all epochs and the final test set eval and metrics
it saves the model as it was after the last epoch
So basically its going to hold a state_dict which is a dictonary PyTorch calls that maps each layer to its weight values
'''
MODEL_SAVE_PATH = "./colon_cancer_model.pth"

# Classes - colon_n is benign and colon_aca is cancerous
CLASSES = ["colon_n", "colon_aca"] # 0 = benign, 1 = cancerous

# Training hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
# This is 0.0001, the learning rate just controls how big a step the optimizer takes, this is a normal safe starting point
LEARNING_RATE = 1e-4
# Every image gets resized to 224x224 pixels since thats the size ResNet18 was pretrained on
IMAGE_SIZE = 224
# Seed makes randomness reproducable, it fixes the starting point so our results are reproducible
SEED = 42

# Split fraction, how much of train_val is being held out for validation
VAL_FRAC = 0.2

# Has PyTorch use a NVIDIA GPU for math if you have one, if not then CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Normalization, ImageNet stats, this is required for pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]