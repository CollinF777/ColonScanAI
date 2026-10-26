# This is the code that teaches our model to recognize images and checks how well it learns
# Torch is the tools for building/training the neural network
import torch
# matplotlib is more plotting tools to create graphs
import matplotlib.pyplot as plt

from config import DEVICE

# This function is for training the model for one complete pass through the training dataset
# model: the neural network (ResNet18) loader: Supplies batches of images and labels
# criterion: calculates how wrong the models predictions are o
# optimizer: adjusts the models weights to improve predictions
def train_one_epoch(model, loader, criterion, optimizer):
    # This just tells PyTorch the model is being trained
    # We do this because some layers behave differently between training and eval so it sets it to training behavior
    model.train()
    '''
    running_loss: tracks the total error across the training images
    correct: counts how many images the model classifies correctly
    total: counts how many images have been processed
    '''
    running_loss, correct, total = 0.0, 0, 0

    # The loader supplies training data in small groups called batches so if we have 1000 images with a batch size of 32, we process 32imgs at a time
    # Images contains the actual pictures and labels contains the correct class for each picture
    for images, labels in loader:
        # This puts the input data on the same device as the model to perform calculations
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # This clears old gradients, PyTorch accumulates gradients so this clears any left over
        optimizer.zero_grad()
        # This is where we make predictions, the images go through the neural network and the model produces predictions
        outputs = model(images)
        # This calculates the loss so we can see how far off the model is
        loss = criterion(outputs, labels)
        '''
        This is where we do backpropagation
        Basically PyTorch now works backword through the network to calculate how the models weights
        contributed to the error
        
        To make it really simple, it figures out how each weight should change to help reduce the loss
        '''
        loss.backward()
        # This is where the training actually happens, the optimizer uses the gradients calculated earlier to adjust the models weights
        optimizer.step()

        # Track the total loss
        running_loss += loss.item() * images.size(0)
        # This returns the highest score and the index of the class with that score
        # We dont need the highest scores though so we only keep the predicated classes
        _, preds = torch.max(outputs, 1)
        # Counts correct predictions
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    # Returns avg loss per image and accuracy as a decimal
    return running_loss / total, correct / total

# This checks how well our model performs on a dataset, for our validation dataset
def evaluate(model, loader, criterion):
    # Just like how we set the model to training behavior in the previous function, this one uses its evaluation behavior
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    # Stores predictions and correct labels so we can analyze the results later
    all_preds, all_labels = [], []

    # During evaluation there isnt any need to calculate gradients since we arent updating weights so this turns it off
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            '''
            This is the part where we saved the predicted classes and correct labels
            .cpu() moves the tensor to the CPU
            In case you guys dont know what a tensor is, its just a data struct that stores numbers
            its basically a container for the data that the model uses
            Its similar to a list but it supports the mathematical operations needed to train neural networks
            .numpy() just converts it to a NumPy array and .extend() adds it to the existing list
            We move it the CPU since NumPy arrays generally cannot be created directly from GPU tensors
            '''
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels

# This function is for running the entire training process
def run_training(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    '''
    This is a dictionary that stores the models performance at the end of every epoch
    train_loss stores the training loss per epoch while train_acc stores the training accuracy per epoch
    val_loss is the validation loss per epoch and val_acc is the validation accuracy per epoch
    '''
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # This just trains the model for the requested number of epochs
    for epoch in range(num_epochs):
        # This trains the model using the training dataset first
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        # This is checking the model with the validation dataset, we ignore prediction and labels since we dont need them to track progress
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train loss: {train_loss:.3f} acc {train_acc:.3f} | "
              f"Val loss: {val_loss:.3f} acc {val_acc:.3f}")

    return history

# This function creates a graph so we can see how the model learned over time
def plot_training_curves(history):
    # Creates a figure with one row and two columns, the rest is pretty self-explanatory
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.show()