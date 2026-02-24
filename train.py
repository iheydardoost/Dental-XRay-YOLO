# =========================
# ===== CONFIGURATION =====
# =========================

# --- Reproducibility ---
seed = 77

# --- Dataset ---
dataset_dir = "../data/a_5_5"
dataset_yaml_path = "../data/dataset.yaml"

# --- Classes ---
# IMPORTANT: order must match your label indices
class_names = [
    "Caries",
    "Crown",
    "Filling",
    "Implant",
    "Bone Loss",
    "Mandibular Canal",
    "Missing teeth",
    "Periapical lesion",
    "maxillary sinus",
    "Root Canal Treatment",
    "Root Piece",
    "impacted tooth"
]

# --- Model ---
# all .pt are pretrained with COCO weights
model_name = "yolov8x.pt"         # size: n / s / m / l / x
img_size = 640

# --- Output ---
project_name = "dental_yolo"
experiment_name = "yolov8x_train_6"

# --- Training ---
epochs = 100
batch_size = 16
learning_rate = 0.005
optimizer = "SGD"         # SGD | Adam | AdamW
workers = 8

# --- Hardware ---
device = "0"                  # 0 / 1 = GPU, "cpu" for CPU
amp = True                  # mixed precision (recommended)

# =========================
# ===== IMPLEMENTATION ====
# =========================

import os
import yaml
import random
import numpy as np
import torch
import albumentations as A
from ultralytics import YOLO

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dataset_yaml():
    """
    Create YOLO dataset YAML file automatically
    """
    data = {
        "path": dataset_dir,
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {i: name for i, name in enumerate(class_names)},
        "nc": len(class_names),
    }

    with open(dataset_yaml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"[INFO] Dataset YAML created at: {dataset_yaml_path}")



def check_gpu():
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available but GPU training requested!")
    print(f"[INFO] Using device: {'GPU' if device != 'cpu' else 'CPU'}")



def train():
    print("[INFO] Loading YOLO model...")
    model = YOLO(model_name)

    # =========================
    # Custom Albumentations
    # =========================

    custom_transforms = [
        A.Blur(blur_limit=7, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.CLAHE(clip_limit=4.0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.CoarseDropout(
            num_holes_range=(1, 5),
            hole_height_range=(10, 40),
            hole_width_range=(10, 40),
            fill_value=0,
            p=0.5
        )
    ]

    print("[INFO] Starting training...")
    model.train(
        data=dataset_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        lr0=learning_rate,
        optimizer=optimizer,
        device=device,
        workers=workers,
        amp=amp,
        project=project_name,
        name=experiment_name,
        seed=seed,
        pretrained=True,
        verbose=True,

        # Inject custom Albumentations
        augmentations=custom_transforms,
    )

    print("[INFO] Training completed!")



if __name__ == "__main__":
    set_seed(seed)
    check_gpu()
    create_dataset_yaml()
    train()
