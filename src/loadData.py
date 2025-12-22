import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import shutil

TRAIN_DIR = "../data/processed/train"
VAL_DIR = "../data/processed/val"
TEST_DIR = "../data/processed/test"

# ✅ Data augmentation only for training dataset
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ✅ Fix weird .DS_STORE Mac folder bug (important on Windows too sometimes)
def clean_ds_store(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):

        for f in filenames:
            if f == ".DS_Store":
                os.remove(os.path.join(dirpath, f))

        for d in dirnames:
            if d == ".DS_Store":
                shutil.rmtree(os.path.join(dirpath, d))


for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    clean_ds_store(folder)


def get_dataloaders(batch_size=4):
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Loading processed datasets...")

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = get_dataloaders()

    print("Classes:", train_dataset.classes)
    print(f"Train images: {len(train_dataset)}")
    print(f"Val images:   {len(val_dataset)}")
    print(f"Test images:  {len(test_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    print("Data loading completed!")





