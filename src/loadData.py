import os
import shutil
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Path Configuration ---
TRAIN_DIR = "../data/processed/train"
VAL_DIR = "../data/processed/val"
TEST_DIR = "../data/processed/test"

# ---- Albumentations wrapper for ImageFolder ----
class AlbuDataset(datasets.ImageFolder):
    def __init__(self, root, albu_transform=None, **kwargs):
        super().__init__(root, **kwargs)
        self.albu_transform = albu_transform

    def __getitem__(self, index):
        path, label = self.samples[index]
        # Open and ensure RGB
        image = np.array(Image.open(path).convert("RGB"))

        if self.albu_transform:
            augmented = self.albu_transform(image=image)
            image = augmented["image"]

        return image, label

# ---- Strong train transform (Forces model to learn leaf texture) ----
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=30, p=0.7),
    # The OneOf block helps the model ignore bad lighting/weather
    A.OneOf([
        A.RandomShadow(p=1),
        A.RandomFog(p=1, fog_coef_lower=0.1, fog_coef_upper=0.2),
        A.RandomRain(p=1),
    ], p=0.4),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.GaussNoise(p=0.3),
    A.Perspective(p=0.3),
    # ImageNet normalization is critical for EfficientNet
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ---- Val/test stays simple (No augmentation) ----
val_test_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def clean_ds_store(root_dir):
    """Removes hidden OS files that can break ImageFolder."""
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f == ".DS_Store" or f.startswith("._"):
                try:
                    os.remove(os.path.join(dirpath, f))
                except:
                    pass

def get_dataloaders(batch_size=16):
    """
    Returns datasets and loaders. 
    Default batch_size 16 is recommended for Laptop CPUs with 8GB-16GB RAM.
    """
    # Clean folders before loading
    for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if os.path.exists(folder):
            clean_ds_store(folder)

    train_dataset = AlbuDataset(root=TRAIN_DIR, albu_transform=train_transform)
    val_dataset   = AlbuDataset(root=VAL_DIR,   albu_transform=val_test_transform)
    test_dataset  = AlbuDataset(root=TEST_DIR,  albu_transform=val_test_transform)

    # num_workers=0 is safer for Windows/Laptop CPUs to avoid Multiprocessing errors
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test block to verify everything is working
    try:
        t_ds, v_ds, ts_ds, t_ld, v_ld, ts_ld = get_dataloaders(batch_size=16)
        print(f"✅ Success! Loaded {len(t_ds)} training images.")
        print(f"Classes found: {t_ds.classes}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")




