import os

BASE = "D:/Major/data/processed"
for split in ["train", "val", "test"]:
    print(f"\n── {split.upper()} ──")
    total = 0
    for cls in sorted(os.listdir(os.path.join(BASE, split))):
        count = len(os.listdir(os.path.join(BASE, split, cls)))
        print(f"  {cls}: {count}")
        total += count
    print(f"  TOTAL: {total}")