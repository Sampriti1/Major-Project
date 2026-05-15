import os

DATASET1 = "D:/Major/data/raw/dataset1"
DATASET2 = "D:/Major/data/raw/dataset2"

def check_dataset(path, name):
    print(f"\n=== {name} ===")
    if not os.path.exists(path):
        print(f"Path not found: {path}")
        return
    
    for cls in os.listdir(path):
        cls_path = os.path.join(path, cls)
        if os.path.isdir(cls_path):
            count = len([
                f for f in os.listdir(cls_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            print(f"{cls}: {count} images")

check_dataset(DATASET1, "DATASET 1")
check_dataset(DATASET2, "DATASET 2")