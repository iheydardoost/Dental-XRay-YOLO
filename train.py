# =========================
# ===== CONFIGURATION =====
# =========================

# --- Reproducibility ---
seed = 42

# --- Dataset ---
dataset_dir = "D:/iman_heydardoost/research_papers/dental_paper_1/data/a_10_10"
dataset_yaml_path = "D:/iman_heydardoost/research_papers/dental_paper_1/data/dataset.yaml"

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
model_name = "yolov8n.pt"         # size: n / s / m / l / x
img_size = 640

# --- Training ---
epochs = 150
batch_size = 4
learning_rate = 0.001
optimizer = "AdamW"         # SGD | Adam | AdamW
workers = 4

# --- Hardware ---
device = "0"                  # 0 / 1 = GPU, "cpu" for CPU
amp = True                  # mixed precision (recommended)

# --- Output ---
project_name = "dental_yolo"
experiment_name = "yolov8_training"

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
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
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
