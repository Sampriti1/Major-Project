import Augmentor
import os
import shutil

RAW_DIR = "D:/Major/data/raw/kaggle/tea_sickness_dataset"

# Only augment the critically low classes
fix_classes = {
    "Anthracnose":   530,   # 70 → 600
    "bird eye spot": 530,   # 70 → 600
    "white spot":    500,   # 99 → 599
}

for cls, sample_count in fix_classes.items():
    cls_path = os.path.join(RAW_DIR, cls)
    if not os.path.isdir(cls_path):
        print(f"NOT FOUND: {cls}")
        continue

    print(f"\nAugmenting {cls} — generating {sample_count} images...")
    p = Augmentor.Pipeline(cls_path)
    p.rotate(probability=0.8, max_left_rotation=25, max_right_rotation=25)
    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.4)
    p.zoom_random(probability=0.6, percentage_area=0.75)
    p.random_brightness(probability=0.7, min_factor=0.5, max_factor=1.5)
    p.random_color(probability=0.7, min_factor=0.5, max_factor=1.5)
    p.random_contrast(probability=0.7, min_factor=0.5, max_factor=1.6)
    p.random_distortion(probability=0.5, grid_width=4,
                        grid_height=4, magnitude=5)
    p.shear(probability=0.3, max_shear_left=10, max_shear_right=10)
    p.sample(sample_count)
    print(f"Done: {cls}")

# Move augmented images out of output/ subfolder
print("\nMoving augmented images...")
for cls in fix_classes.keys():
    cls_path = os.path.join(RAW_DIR, cls)
    output_path = os.path.join(cls_path, "output")
    if not os.path.exists(output_path):
        continue
    moved = 0
    for img in os.listdir(output_path):
        if img.lower().endswith(('.jpg', '.jpeg', '.png')):
            shutil.move(
                os.path.join(output_path, img),
                os.path.join(cls_path, img)
            )
            moved += 1
    os.rmdir(output_path)
    print(f"  {cls}: moved {moved} augmented images")

# Verify final counts
print("\nFinal RAW counts:")
total = 0
for cls in sorted(os.listdir(RAW_DIR)):
    cls_path = os.path.join(RAW_DIR, cls)
    if os.path.isdir(cls_path):
        count = len([f for f in os.listdir(cls_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        status = "✅" if count >= 500 else "⚠️"
        print(f"  {cls}: {count} {status}")
        total += count
print(f"\nTOTAL: {total} images")
print("\nNow run: preprocess.py → train.py")