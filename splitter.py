import os
import shutil
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# Set parameters (customize as needed)
seed = 10  # For reproducibility
test_percent = 10
valid_percent = 10
dataset_dir = '../data/original'  # Root directory
target_dir = '../data/a_10_10'
default_max_workers = 4  # For ThreadPoolExecutor; adjust based on CPU

# Paths to original images and labels
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')

# Get list of image files (assuming common extensions like .jpg, .png, .jpeg)
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# Shuffle with seed
random.seed(seed)
random.shuffle(image_files)

# Calculate split sizes
total_images = len(image_files)
test_size = int(total_images * (test_percent / 100))
valid_size = int(total_images * (valid_percent / 100))
train_size = total_images - test_size - valid_size

# Split the lists
train_images = image_files[ : train_size]
valid_images = image_files[train_size : train_size + valid_size]
test_images = image_files[train_size + valid_size : ]

# Create directories for splits
os.makedirs(target_dir, exist_ok=True)
for split in ['train', 'valid', 'test']:
    os.makedirs(os.path.join(target_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, split, 'labels'), exist_ok=True)

# Function to copy a single file pair (image and label)
def copy_file_pair(img_file, split):
    try:
        # Copy image
        src_img = os.path.join(images_dir, img_file)
        dst_img = os.path.join(target_dir, split, 'images', img_file)
        shutil.copy(src_img, dst_img)
        
        # Copy matching label (assume label filename is image basename + .txt)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        src_label = os.path.join(labels_dir, label_file)
        if os.path.exists(src_label):
            dst_label = os.path.join(target_dir, split, 'labels', label_file)
            shutil.copy(src_label, dst_label)
        else:
            print(f"Warning: Label not found for {img_file}", file=sys.stderr)
        return f"Copied {img_file}"
    except Exception as e:
        print(f"Error copying {img_file}: {e}", file=sys.stderr)
        return None

# Function to copy files in parallel with progress (using threads for I/O bound task)
def copy_files_parallel(image_list, split, max_workers=default_max_workers):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_img = {executor.submit(copy_file_pair, img, split): img for img in image_list}
        for future in tqdm(as_completed(future_to_img), total=len(image_list), desc=f"Copying {split} files"):
            pass  # Results handled in function; just for progress

if __name__ == '__main__':
    try:
        copy_files_parallel(train_images, 'train')
        copy_files_parallel(valid_images, 'valid')
        copy_files_parallel(test_images, 'test')
        print(f"Split complete: Train={len(train_images)}, Valid={len(valid_images)}, Test={len(test_images)}")
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")
        sys.exit(0)