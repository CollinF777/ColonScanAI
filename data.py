'''
Cameron this is your part
This is for dataset loading
Basically file listing, train/validation splits, PyTorch Datasets, and image transforms are going to be in here

I did the imports youll need and check the comments as Im trying to be very thorough with what we need
You should use the comments as a point of reference and you can delete them when you've done each section
'''
# This is for getting the files
import os
# Youll use random to split the images into training and validation groups randomly
import random
# PIL is an Imaging library, Image is going to let you open and manipulate files
from PIL import Image

'''
So you are going to make a dataset in the file 
A dataset is a collection of examples the model can learn from
So it will contain an image and its correct label for ours
'''
from torch.utils.data import Dataset
'''
Transforms are very important for this file
They provide the tools for changing images into the format the model is exprecting
This is important because our model cant just recieve a filename, it needs the image data in a numerical format
'''
from torchvision import transforms

from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

'''
You should first have a class ColonCancerDataset(Dataset) here
Its mean to load the file paths and labels + apply transforms on access
The class should have three methods
def __init__ which has params file_paths, labels, and transform=None. This is just a constructor
def __len__ which should just return length of the number of file_paths
def__getitem__ which has param idx where idx is the index of the item you want, it reterives one item from the dataset
__getitem__ should open the file and convert to RGB, get the label, and transform if needed. It should return the image and label
'''

'''
After you should have a function build_file_list which has params data_dir and classes
It should walk the dataset folder and return file_paths and labels
It should only read folders that match our classes, nothing extra
It should also give each class a numerical label label_idx user enumerate when looping
It should find the current class folder, then list all the files in that folder and append them to our file_paths, the label should also be saved
After all that it should return the file_paths and labels
'''

'''
Up next is a function called something like train_val_split 
It would have params file_paths, labels, and val_frac=0.2
This function should split one combined list into training and validation sets
you should make a combined variable from the file_paths and labels then shuffle them to make it random
After that you should then calculate the number of validation images and create the validation set as well as the training set
Make sure you also make a helper unzip inner function that converts the list of image-label pairs back to separate lists and 
return the unzipped training and validation sets
'''

'''
Finally down here you should do your transformations 
The training should get some light augmentation to help the model generalize
Validation set should only get resized and normalized

So the train transform should be resized, then have a random horizonal flip and random rotation
then it can be made a Tensor and normalized with the mean and std

The eval transform should just be resized, made a tensor, and normalized
'''