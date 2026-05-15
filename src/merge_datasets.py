import os
import shutil

# ── PATHS ───────────────────────────────────────────────
EXISTING = "D:/Major/data/raw/kaggle/tea_sickness_dataset"
DATASET1 = "D:/Major/data/raw/dataset1"
DATASET2 = "D:/Major/data/raw/dataset2"
# ───────────────────────────────────────────────────────

# ── YOUR FINAL CLASSES (DO NOT CHANGE) ──────────────────
YOUR_CLASSES = [
    'algal leaf', 'Anthracnose', 'bird eye spot',
    'brown blight', 'gray light', 'healthy',
    'red leaf spot', 'white spot'
]

# ── DATASET 1 MAPPING ──────────────────────────────────
DATASET1_MAP = {
    "algal_spot": "algal leaf",
    "brown_blight": "brown blight",
    "gray_blight": "gray light",
    "healthy": "healthy",
    "red_spot": "red leaf spot",
    # "helopeltis": skipped
}

# ── DATASET 2 MAPPING ──────────────────────────────────
DATASET2_MAP = {
    "1. Tea algal leaf spot": "algal leaf",
    "2. Brown Blight": "brown blight",
    "3. Gray Blight": "gray light",
    "7. Healthy leaf": "healthy",
    # skipped: helopeltis, red spider, green mirid bug
}

# ── COPY FUNCTION ──────────────────────────────────────
def copy_images(src_folder, dst_folder, prefix):
    os.makedirs(dst_folder, exist_ok=True)
    count = 0

    for img in os.listdir(src_folder):
        if img.lower().endswith(('.jpg', '.jpeg', '.png')):
            src = os.path.join(src_folder, img)
            new_name = f"{prefix}_{img}"
            dst = os.path.join(dst_folder, new_name)

            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                count += 1

    return count


# ── MERGE FUNCTION ─────────────────────────────────────
def merge_dataset(dataset_path, class_map, prefix, name):
    print(f"\n===== Merging {name} =====")
    total = 0

    for folder in os.listdir(dataset_path):
        src_folder = os.path.join(dataset_path, folder)

        if not os.path.isdir(src_folder):
            continue

        if folder in class_map:
            target_class = class_map[folder]
            dst_folder = os.path.join(EXISTING, target_class)

            copied = copy_images(src_folder, dst_folder, prefix)
            total += copied

            print(f"{folder} → {target_class}: +{copied}")

        else:
            print(f"SKIPPED: {folder}")

    print(f"Total from {name}: {total}")


# ── RUN MERGE ──────────────────────────────────────────
merge_dataset(DATASET1, DATASET1_MAP, "d1", "DATASET 1")
merge_dataset(DATASET2, DATASET2_MAP, "d2", "DATASET 2")


# ── FINAL COUNT ────────────────────────────────────────
print("\n===== FINAL DATASET COUNT =====")
grand_total = 0

for cls in sorted(os.listdir(EXISTING)):
    cls_path = os.path.join(EXISTING, cls)

    if os.path.isdir(cls_path):
        count = len([
            f for f in os.listdir(cls_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        print(f"{cls}: {count}")
        grand_total += count

print(f"\nTOTAL IMAGES: {grand_total}")
print("\n✅ Merge completed successfully!")