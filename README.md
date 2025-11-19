# Dental-XRay-YOLO
some scripts to preprocess dataset of dental xray images and training YOLO models for them. detection of dental issues is our concern.

## splitter.py
this script is used to split the dataset into train, valid, test directories. set parameters at first of script and use.

## augmentation.py
this script applies augmentations on splitted dataset. set parameters at first of script and use.  
available augmentations for now:  
*  'horizontal_flip',  
*  'vertical_flip',  
*  'rotation',  
*  'brightness',  
*  'contrast',  
*  'color_jitter',  
*  'gaussian_noise',  
*  'gaussian_blur',  
*  'clahe',  
*  'random_gamma',  
*  'elastic_transform',  

# To Do
* adding training script which uses GPU to train YOLO model