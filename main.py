import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json

import config
from data import ColonCancerDataset, build_file_list, train_val_split, train_transform, eval_transform
from model import build_model
from train import run_training, evaluate, plot_training_curves
from evaluate import ClinicalEvaluator
from gradcam import show_gradcam_samples

# False = reuse saved model (skip training), True = retrain from scratch (do this after you change the model, data, or anything like that)
RETRAIN = False
# Saved so training curves can be re-plotted without retraining
HISTORY_PATH = "./training_history.json"

def main():
    print(f"Using device: {config.DEVICE}")
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    '''
    This is where we handle the data
    Train and validation come from one folder and get split here
    The test set is in its own folder and isnt ever used for training
    '''
    trainval_paths, trainval_labels = build_file_list(config.TRAINVAL_DIR, config.CLASSES)

    train_paths, train_labels, val_paths, val_labels = train_val_split(trainval_paths, trainval_labels, config.VAL_FRAC)
    test_paths, test_labels = build_file_list(config.TEST_DIR, config.CLASSES)
    print(f"Set sizes | Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

    train_ds = ColonCancerDataset(train_paths, train_labels, train_transform)
    val_ds = ColonCancerDataset(val_paths, val_labels, eval_transform)
    test_ds = ColonCancerDataset(test_paths, test_labels, eval_transform)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=True)

    '''
    This is where we handle the model
    We start with building it then training it before going on to eval
    If the model has already been trained then we just use that unless otherwise specified since 
    it takes a pretty long time to train
    '''
    model = build_model(num_classes=len(config.CLASSES), freeze_backbone=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=config.LEARNING_RATE)

    if os.path.exists(config.MODEL_SAVE_PATH) and not RETRAIN:
        print(f"Found {config.MODEL_SAVE_PATH}, loading saved weights (training skipped).")
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH) as f:
                plot_training_curves(json.load(f))
    else:
        history = run_training(model, train_loader, val_loader, criterion, optimizer, config.NUM_EPOCHS)
        plot_training_curves(history)

        # Save weights and history so you dont need to train every time
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
        print(f"Model saved to {config.MODEL_SAVE_PATH}")
        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f)

    # Final test eval
    test_loss, test_acc, preds, true_labels = evaluate(model, test_loader, criterion)
    print(f"\nTest accuracy: {test_acc:.3f}")
    # pos_label defaults to 1  (colon_aca/cancerous), so recall is cancel recall
    ClinicalEvaluator().print_clinical_metrics(true_labels, preds, config.CLASSES)

    # Save the model
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Model saved to {config.MODEL_SAVE_PATH}")

    # Grad CAM
    print("\nGenerating Grad-CAM visual")
    show_gradcam_samples(model, test_paths, test_labels, num_samples=3)

if __name__ == '__main__':
    main()