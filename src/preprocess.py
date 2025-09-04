import os
import shutil
import random
from pathlib import Path
from PIL import Image


RAW_DIR = Path("../data/raw/kaggle/tea_sickness_dataset")  
PROCESSED_DIR = Path("../data/processed")


TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_SIZE = (224, 224) 


def create_dirs():
    """Create processed/train, val, test folders."""
    for split in ["train", "val", "test"]:
        for cls in os.listdir(RAW_DIR):
            (PROCESSED_DIR / split / cls).mkdir(parents=True, exist_ok=True)


def resize_and_save(src_path, dest_path):
    """Resize image and save to destination."""
    try:
        img = Image.open(src_path).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        img.save(dest_path)
    except Exception as e:
        print(f" Error processing {src_path}: {e}")


def split_and_process():
    """Split dataset and copy resized images into processed folder."""
    for cls in os.listdir(RAW_DIR):
        cls_path = RAW_DIR / cls
        if not cls_path.is_dir():
            continue

        images = list(cls_path.glob("*.*"))
        random.shuffle(images)

        total = len(images)
        train_end = int(TRAIN_RATIO * total)
        val_end = train_end + int(VAL_RATIO * total)

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        for split, files in splits.items():
            for src in files:
                dest = PROCESSED_DIR / split / cls / src.name
                resize_and_save(src, dest)

        print(f" {cls}: {total} images → "
              f"{len(splits['train'])} train, "
              f"{len(splits['val'])} val, "
              f"{len(splits['test'])} test")


if __name__ == "__main__":
    print(" Starting preprocessing...")
    create_dirs()
    split_and_process()
    print(" Preprocessing completed! Data saved in data/processed/")
