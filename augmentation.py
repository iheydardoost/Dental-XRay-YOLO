import os
import sys
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import albumentations as A
from PIL import Image
import shutil
import cv2

# Configuration: Set all parameters, paths, and seed here
seed = 77  # For reproducibility
random.seed(seed)
np.random.seed(seed)

input_dataset_dir = 'D:/iman_heydardoost/research_papers/dental_paper_1/data/a_10_10'  # Input split dataset directory
output_dataset_dir = 'D:/iman_heydardoost/research_papers/dental_paper_1/data/a_10_10_aug'  # Output augmented directory
aug_samples_dir = 'D:/iman_heydardoost/research_papers/dental_paper_1/data/a_10_10_aug_samples'  # Directory for sample original and augmented images
max_workers = 8  # For ThreadPoolExecutor; adjust based on CPU

# List of augmentations to apply (comment out to disable modularity)
selected_augs = [
    'horizontal_flip',
    'vertical_flip',
    'rotation',
    'brightness',
    'contrast',
    'color_jitter',
    'gaussian_noise',
    'gaussian_blur',
    'clahe',
    'random_gamma',
    'elastic_transform',
]

geometric_augs = ['horizontal_flip', 'vertical_flip', 'rotation', 'elastic_transform']

# --- Horizontal Flip parameters ---
horizontal_flip_p = 0.5

# --- Vertical Flip parameters ---
vertical_flip_p = 0.5

# --- Rotation parameters ---
rotation_limit = 10.0  # ± degrees
rotation_p = 0.5

# --- Brightness parameters ---
brightness_limit = 0.2  # ± factor
brightness_p = 0.5

# --- Contrast parameters ---
contrast_limit = 0.2  # ± factor
contrast_p = 0.5

# --- Color Jitter (HueSaturationValue) parameters ---
color_jitter_hue_shift_limit = 20
color_jitter_sat_shift_limit = 30
color_jitter_val_shift_limit = 20
color_jitter_p = 0.3

# --- GaussianNoise parameters (for machine noise in X-rays) ---
gaussian_noise_mean_range = (-0.1, 0.1)
gaussian_noise_std_range = (0.1, 0.3)
gaussian_noise_p = 0.2

# --- GaussianBlur parameters (for low-contrast contours) ---
gaussian_blur_blur_limit = (3, 7)
gaussian_blur_p = 0.5

# --- CLAHE parameters (contrast enhancement for X-rays) ---
clahe_clip_limit = (1.0, 4.0)
clahe_tile_grid_size = (8, 8)
clahe_p = 0.5

# --- RandomGamma parameters (intensity adjustments) ---
random_gamma_gamma_limit = (80, 120)
random_gamma_p = 0.5

# --- ElasticTransform parameters (anatomical variations/deformations) ---
elastic_transform_alpha = 1
elastic_transform_sigma = 50
elastic_transform_p = 0.5

# Modular augmentation functions (each returns an Albumentations transform)
def get_horizontal_flip_transform():
    return A.HorizontalFlip(p=horizontal_flip_p)

def get_vertical_flip_transform():
    return A.VerticalFlip(p=vertical_flip_p)

def get_rotation_transform():
    return A.Rotate(limit=rotation_limit, p=rotation_p)

def get_brightness_transform():
    return A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=0,
                                      p=brightness_p)

def get_contrast_transform():
    return A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=contrast_limit,
                                      p=contrast_p)

def get_color_jitter_transform():
    return A.HueSaturationValue(hue_shift_limit=color_jitter_hue_shift_limit,
                                sat_shift_limit=color_jitter_sat_shift_limit,
                                val_shift_limit=color_jitter_val_shift_limit,
                                p=color_jitter_p)

def get_gaussian_noise_transform():
    return A.GaussNoise(mean_range=gaussian_noise_mean_range, std_range=gaussian_noise_std_range, p=gaussian_noise_p)

def get_gaussian_blur_transform():
    return A.GaussianBlur(blur_limit=gaussian_blur_blur_limit, p=gaussian_blur_p)

def get_clahe_transform():
    return A.CLAHE(clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_grid_size, p=clahe_p)

def get_random_gamma_transform():
    return A.RandomGamma(gamma_limit=random_gamma_gamma_limit, p=random_gamma_p)

def get_elastic_transform_transform():
    return A.ElasticTransform(alpha=elastic_transform_alpha, sigma=elastic_transform_sigma, p=elastic_transform_p)

# Build Compose based on selected augs
def build_transforms():
    transforms = []
    has_geometric = any(aug in geometric_augs for aug in selected_augs)
    if 'horizontal_flip' in selected_augs:
        transforms.append(get_horizontal_flip_transform())
    if 'vertical_flip' in selected_augs:
        transforms.append(get_vertical_flip_transform())
    if 'rotation' in selected_augs:
        transforms.append(get_rotation_transform())
    if 'brightness' in selected_augs:
        transforms.append(get_brightness_transform())
    if 'contrast' in selected_augs:
        transforms.append(get_contrast_transform())
    if 'color_jitter' in selected_augs:
        transforms.append(get_color_jitter_transform())
    if 'gaussian_noise' in selected_augs:
        transforms.append(get_gaussian_noise_transform())
    if 'gaussian_blur' in selected_augs:
        transforms.append(get_gaussian_blur_transform())
    if 'clahe' in selected_augs:
        transforms.append(get_clahe_transform())
    if 'random_gamma' in selected_augs:
        transforms.append(get_random_gamma_transform())
    if 'elastic_transform' in selected_augs:
        transforms.append(get_elastic_transform_transform())
    
    bbox_params = A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=0, min_visibility=0) if has_geometric else None
    keypoint_params = A.KeypointParams(format='xy', label_fields=['keypoint_object_ids'], remove_invisible=True) if has_geometric else None
    
    return A.Compose(transforms, bbox_params=bbox_params, keypoint_params=keypoint_params)

# Load YOLO labels: [class, x_center, y_center, width, height, (segments: x1 y1 x2 y2 ...)]
def load_labels(label_path):
    classes = []
    bboxes = []
    segments = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            l = line.strip().split()
            if not l:
                continue
            try:
                label = list(map(float, l))
                if len(label) < 5:
                    continue  # Invalid, skip
                cls = int(label[0])
                box = label[1:5]
                seg = label[5:] if len(label) > 5 else []
                if len(seg) % 2 != 0:
                    print(f"Invalid segment length (odd) in {label_path}: {len(seg)}")
                    continue
                # Clean box to [0,1]
                x, y, w, h = box
                w = min(w, 1.0)
                h = min(h, 1.0)
                x_min = max(0.0, x - w / 2)
                x_max = min(1.0, x + w / 2)
                y_min = max(0.0, y - h / 2)
                y_max = min(1.0, y + h / 2)
                w = x_max - x_min
                h = y_max - y_min
                x = (x_min + x_max) / 2
                y = (y_min + y_max) / 2
                box = [x, y, w, h]
                # Clean segments to [0,1]
                for i in range(0, len(seg), 2):
                    seg[i] = max(0.0, min(1.0, seg[i]))  # x
                    seg[i + 1] = max(0.0, min(1.0, seg[i + 1]))  # y
                classes.append(cls)
                bboxes.append(box)
                segments.append(seg)
            except ValueError:
                print(f"Invalid values in line of {label_path}: {line}")
    return classes, bboxes, segments

# Save YOLO labels
def save_labels(classes, bboxes, segments, label_path):
    with open(label_path, 'w') as f:
        for cls, bbox, seg in zip(classes, bboxes, segments):
            line = [cls] + bbox + seg
            f.write(' '.join(map(str, line)) + '\n')

# Process single image: Apply transforms, save augmented
def process_image(img_file, train_images_dir, train_labels_dir, aug_train_images_dir, aug_train_labels_dir, transform):
    try:
        img_path = os.path.join(train_images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(train_labels_dir, label_file)
        
        image = np.asarray(Image.open(img_path))
        classes, bboxes, segments = load_labels(label_path)
        
        flat_keypoints = []
        keypoint_object_ids = []
        for obj_id, seg in enumerate(segments):
            for j in range(0, len(seg), 2):
                flat_keypoints.append((seg[j], seg[j+1]))
                keypoint_object_ids.append(obj_id)
        
        augmented = transform(image=image, bboxes=bboxes, class_labels=classes, keypoints=flat_keypoints, keypoint_object_ids=keypoint_object_ids)
        
        aug_image = Image.fromarray(augmented['image'])
        aug_classes = augmented['class_labels']
        aug_bboxes = augmented['bboxes']
        aug_keypoints = augmented['keypoints']
        aug_keypoint_object_ids = augmented['keypoint_object_ids']
        
        aug_segments = [[] for _ in aug_bboxes]
        for kp_id, kp in zip(aug_keypoint_object_ids, aug_keypoints):
            aug_segments[int(kp_id)].extend(kp)
        
        aug_img_path = os.path.join(aug_train_images_dir, img_file)
        aug_label_path = os.path.join(aug_train_labels_dir, label_file)
        
        aug_image.save(aug_img_path)
        if aug_bboxes:
            save_labels(aug_classes, aug_bboxes, aug_segments, aug_label_path)
        
        return f"Processed {img_file}"
    except Exception as e:
        print(f"Error processing {img_file}: {e}", file=sys.stderr)
        return None

# Create visualization samples for each augmentation on one image (independent function)
def create_aug_samples(train_images_dir, train_labels_dir):
    os.makedirs(aug_samples_dir, exist_ok=True)
    os.makedirs(os.path.join(aug_samples_dir, 'labels'), exist_ok=True)
    
    # Pick first image as sample
    image_files = [f for f in os.listdir(train_images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not image_files:
        print("No images found for sampling.")
        return
    sample_img_file = image_files[0]
    sample_img_path = os.path.join(train_images_dir, sample_img_file)
    sample_label_file = os.path.splitext(sample_img_file)[0] + '.txt'
    sample_label_path = os.path.join(train_labels_dir, sample_label_file)
    
    sample_image = np.asarray(Image.open(sample_img_path))
    sample_classes, sample_bboxes, sample_segments = load_labels(sample_label_path)
    
    # Save original
    original_save_path = os.path.join(aug_samples_dir, f'original_{sample_img_file}')
    Image.fromarray(sample_image).save(original_save_path)
    if sample_bboxes:
        save_labels(sample_classes, sample_bboxes, sample_segments, os.path.join(aug_samples_dir, 'labels', f'original_{sample_label_file}'))
    
    # Define max-effect transforms for each aug (p=1.0, max values)
    max_aug_dict = {
        'horizontal_flip': A.Compose([A.HorizontalFlip(p=1.0)], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=0, min_visibility=0),
                                     keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_object_ids'], remove_invisible=True)),
        'vertical_flip': A.Compose([A.VerticalFlip(p=1.0)], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=0, min_visibility=0),
                                   keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_object_ids'], remove_invisible=True)),
        'rotation': A.Compose([A.Rotate(limit=rotation_limit, p=1.0)], 
                              bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=0, min_visibility=0),
                              keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_object_ids'], remove_invisible=True)),
        'brightness': A.Compose([A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=0, p=1.0)], 
                                bbox_params=None, keypoint_params=None),
        'contrast': A.Compose([A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=contrast_limit, p=1.0)], 
                              bbox_params=None, keypoint_params=None),
        'color_jitter': A.Compose([A.HueSaturationValue(hue_shift_limit=color_jitter_hue_shift_limit, sat_shift_limit=color_jitter_sat_shift_limit, 
                                                        val_shift_limit=color_jitter_val_shift_limit, p=1.0)], 
                                  bbox_params=None, keypoint_params=None),
        'gaussian_noise': A.Compose([A.GaussNoise(mean_range=gaussian_noise_mean_range, std_range=gaussian_noise_std_range, p=1.0)], 
                                    bbox_params=None, keypoint_params=None),
        'gaussian_blur': A.Compose([A.GaussianBlur(blur_limit=gaussian_blur_blur_limit, p=1.0)], 
                                   bbox_params=None, keypoint_params=None),
        'clahe': A.Compose([A.CLAHE(clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_grid_size, p=1.0)], 
                           bbox_params=None, keypoint_params=None),
        'random_gamma': A.Compose([A.RandomGamma(gamma_limit=random_gamma_gamma_limit, p=1.0)], 
                                  bbox_params=None, keypoint_params=None),
        'elastic_transform': A.Compose([A.ElasticTransform(alpha=elastic_transform_alpha, sigma=elastic_transform_sigma, p=1.0)], 
                                      bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=0, min_visibility=0),
                                      keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_object_ids'], remove_invisible=True)),
    }
    
    # Apply and save each individual max-effect augmentation
    for aug_name in selected_augs:
        single_transform = max_aug_dict.get(aug_name)
        if single_transform is None:
            continue
        
        sample_flat_keypoints = []
        sample_keypoint_object_ids = []
        for obj_id, seg in enumerate(sample_segments):
            for j in range(0, len(seg), 2):
                sample_flat_keypoints.append((seg[j], seg[j+1]))
                sample_keypoint_object_ids.append(obj_id)
        
        aug = single_transform(image=sample_image, bboxes=sample_bboxes, class_labels=sample_classes, keypoints=sample_flat_keypoints, keypoint_object_ids=sample_keypoint_object_ids)
        
        aug_img = Image.fromarray(aug['image'])
        aug_classes = aug['class_labels']
        aug_bboxes = aug['bboxes']
        aug_keypoints = aug['keypoints']
        aug_keypoint_object_ids = aug['keypoint_object_ids']
        
        aug_segments = [[] for _ in aug_bboxes]
        for kp_id, kp in zip(aug_keypoint_object_ids, aug_keypoints):
            aug_segments[int(kp_id)].extend(kp)
        
        aug_save_path = os.path.join(aug_samples_dir, f'{aug_name}_aug_{sample_img_file}')
        aug_img.save(aug_save_path)
        if aug_bboxes:
            save_labels(aug_classes, aug_bboxes, aug_segments, os.path.join(aug_samples_dir, 'labels', f'{aug_name}_aug_{sample_label_file}'))

    print("Augmentation samples created in aug_samples folder.")

def main():
    # Create output directories
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_dataset_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dataset_dir, split, 'labels'), exist_ok=True)

    # Copy valid and test unchanged
    for split in ['valid', 'test']:
        shutil.copytree(os.path.join(input_dataset_dir, split, 'images'),
                        os.path.join(output_dataset_dir, split, 'images'), dirs_exist_ok=True)
        shutil.copytree(os.path.join(input_dataset_dir, split, 'labels'),
                        os.path.join(output_dataset_dir, split, 'labels'), dirs_exist_ok=True)

    # Augment train set
    train_images_dir = os.path.join(input_dataset_dir, 'train', 'images')
    train_labels_dir = os.path.join(input_dataset_dir, 'train', 'labels')
    aug_train_images_dir = os.path.join(output_dataset_dir, 'train', 'images')
    aug_train_labels_dir = os.path.join(output_dataset_dir, 'train', 'labels')

    # Creating augmented samples
    create_aug_samples(train_images_dir, train_labels_dir)

    image_files = [f for f in os.listdir(train_images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    transform = build_transforms()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_img = {executor.submit(process_image, img, train_images_dir, train_labels_dir,
                                         aug_train_images_dir, aug_train_labels_dir, transform): img
                         for img in image_files}
        for future in tqdm(as_completed(future_to_img), total=len(image_files), desc="Augmenting train files"):
            pass  # Progress; errors handled in func

    print("Augmentation complete!")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")
        sys.exit(0)