import os
import shutil
import random
import time
from pathlib import Path
from PIL import Image
from rembg import remove
from io import BytesIO

# --- Configuration ---
RAW_DIR = Path("../data/raw/kaggle/tea_sickness_dataset")  
PROCESSED_DIR = Path("../data/processed")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
IMAGE_SIZE = (224, 224) 

def create_dirs():
    """Create processed/train, val, test folders."""
    if PROCESSED_DIR.exists():
        print(f"Cleaning existing directory: {PROCESSED_DIR}...")
        shutil.rmtree(PROCESSED_DIR)
        
    for split in ["train", "val", "test"]:
        for cls in os.listdir(RAW_DIR):
            cls_path = RAW_DIR / cls
            if cls_path.is_dir():
                (PROCESSED_DIR / split / cls).mkdir(parents=True, exist_ok=True)

def remove_bg_and_resize(src_path, dest_path):
    """Removes background, resizes, and saves as RGB."""
    try:
        with open(src_path, "rb") as f:
            input_data = f.read()
        
        # This is the heavy part that causes the "pause"
        subject_only_data = remove(input_data)
        
        img = Image.open(BytesIO(subject_only_data)).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        img.save(dest_path)
        return True
    except Exception as e:
        print(f"\n[!] Error on {src_path.name}: {e}")
        return False

def split_and_process():
    """Process images with intervals to let CPU breathe."""
    for cls in os.listdir(RAW_DIR):
        cls_path = RAW_DIR / cls
        if not cls_path.is_dir(): continue

        images = list(cls_path.glob("*.[jJ][pP]*[gG]")) + list(cls_path.glob("*.[pP][nN][gG]"))
        random.shuffle(images)

        total = len(images)
        if total == 0: continue
            
        train_end = int(TRAIN_RATIO * total)
        val_end = train_end + int(VAL_RATIO * total)

        splits = {"train": images[:train_end], "val": images[train_end:val_end], "test": images[val_end:]}

        print(f"\n🚀 Processing Class: {cls}")
        for split, files in splits.items():
            print(f"  -> Split: {split} ({len(files)} images)")
            for count, src in enumerate(files):
                dest = PROCESSED_DIR / split / cls / src.name
                
                # If file already exists, skip it (helps if you have to restart)
                if not dest.exists():
                    remove_bg_and_resize(src, dest)
                
                # UPDATE: Every 5 images, print progress and pause for 1 second
                # This prevents the CPU from locking up the whole system.
                if (count + 1) % 5 == 0:
                    print(f"     Progress: {count + 1}/{len(files)} processed...")
                    time.sleep(1) 

if __name__ == "__main__":
    if not RAW_DIR.exists():
        print(f"Error: RAW_DIR not found at {RAW_DIR}")
    else:
        create_dirs()
        split_and_process()
        print("\n✅ DONE! You can now run train.py.")