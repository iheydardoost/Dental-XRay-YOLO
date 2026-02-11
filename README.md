# Dental-XRay-YOLO
some scripts to preprocess dataset of dental xray images and training YOLO models for them. detection of dental issues is our concern.

## splitter.py
this script is used to split the dataset into train, valid, test directories. set parameters at first of script and use.

## augmentation.py
(DEPRECATED: ultralytics yolo training model support various albumentations, 
so we don't need to do it explicitely)  
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
*  

## train.py
this script is for training yolo model. set the parameters at the first of script properly, 
choose desired YOLO model and train on (CPU/GPU).  
albumentations are available to choose and set in this script.  
list of available transforms: https://explore.albumentations.ai/

# To Do
* let's train and gather data :) .