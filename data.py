# data loading - file listing, train/val split, dataset, transforms
import os
import random
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, SEED

# skip non image files like .DS_Store
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

#-----

class ColonCancerDataset(Dataset):
    # holds paths + labels, images only opened when accessed

    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # force rgb so every image has 3 channels
        # with closes the file, avoids too many open files over long runs
        with Image.open(self.file_paths[idx]) as img:
            image = img.convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

#-----

def build_file_list(data_dir, classes):
    # only reads folders in classes so lung folders get ignored
    # label = index in classes (colon_n 0, colon_aca 1)
    file_paths, labels = [], []

    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)

        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"missing folder: {class_dir}, check data layout")

        # sorted so order is same on every machine
        for file_name in sorted(os.listdir(class_dir)):
            if file_name.lower().endswith(IMAGE_EXTENSIONS):
                file_paths.append(os.path.join(class_dir, file_name))
                labels.append(label_idx)

    return file_paths, labels

#-----

def train_val_split(file_paths, labels, val_frac=0.2, seed=SEED):
    # pair paths w labels so they stay together during shuffle
    combined = list(zip(file_paths, labels))
    # seeded so everyone gets same split
    rng = random.Random(seed)

    # stratified - split each class separately so val has same class ratio as train
    train_set, val_set = [], []
    for label in sorted(set(labels)):
        class_pairs = [pair for pair in combined if pair[1] == label]
        rng.shuffle(class_pairs)
        n_val = int(len(class_pairs) * val_frac)
        val_set.extend(class_pairs[:n_val])
        train_set.extend(class_pairs[n_val:])

    # reshuffle so classes arent in blocks
    rng.shuffle(train_set)
    rng.shuffle(val_set)

    def unzip(pairs):
        # list of (path, label) back to 2 lists
        paths = [path for path, _ in pairs]
        pair_labels = [label for _, label in pairs]
        return paths, pair_labels

    train_paths, train_labels = unzip(train_set)
    val_paths, val_labels = unzip(val_set)

    return train_paths, train_labels, val_paths, val_labels

#-----

# train gets light augmentation to help generalize
# tissue has no up/down so flips + 90 deg turns are free, no black corners like small angles leave
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomChoice([transforms.RandomRotation((angle, angle)) for angle in (0, 90, 180, 270)]),
    # stain color varies a lot slide to slide (pale pink vs deep purple), mild jitter so model learns structure not stain
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# val/test just resize + normalize, no randomness
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
