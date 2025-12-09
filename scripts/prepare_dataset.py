import os
import shutil
from sklearn.model_selection import train_test_split

# Automatically detect project root folder
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(PROJECT_ROOT, "dataset", "train")
VAL_PATH = os.path.join(PROJECT_ROOT, "dataset", "val")

os.makedirs(VAL_PATH, exist_ok=True)

classes = os.listdir(TRAIN_PATH)

for cls in classes:
    os.makedirs(os.path.join(VAL_PATH, cls), exist_ok=True)
    files = os.listdir(os.path.join(TRAIN_PATH, cls))
    train_files, val_files = train_test_split(files, test_size=0.2, random_state=42)

    for f in val_files:
        shutil.move(os.path.join(TRAIN_PATH, cls, f), os.path.join(VAL_PATH, cls, f))

print("Validation split done!")
