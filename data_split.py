import os
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

# Configuration
root_dir = '/home/super/Desktop/bekhzod/backup/tile_classification/25_05_09'  # Original dataset root
# output_dir = 'split_data'; os.makedirs(output_dir, exist_ok=True) # New output directory

rest_ratio, val_ratio, seed = 0.4, 0.2, 2025  # Ratio of remaining images for validation (test gets 1-val_ratio)
# Create output directory structure

for dataset in ['LBC', 'PAP']:
    for cls in ['normal', 'abnormal']:
        src_path = os.path.join(root_dir, dataset, cls)
        if not os.path.exists(src_path): continue
            
        all_images = [f for f in os.listdir(src_path) if f.endswith('.png')]
        core_train = [img for img in all_images if '_180' in img]
        others     = [img for img in all_images if '_180' not in img]
        
        # Single stratified split for remaining images
        train_extra, remaining = train_test_split(others, test_size=rest_ratio, random_state=seed, stratify=[cls]*len(others))
        
        # Split remaining into val/test
        val_images, test_images = train_test_split(remaining, test_size=val_ratio, random_state=seed, stratify=[cls]*len(remaining))
        
        # Combine core + extra training images
        train_images = core_train + train_extra

        print(f'{dataset} Split completed for {cls} images: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test per class')
        
        
        # print(f"Train images -> {len(train_images)}")
        # print(f"Valid images -> {len(val_images)}")
        # print(f"Test images -> {len(test_images)}")

        # Copy files
        # for split, imgs in [('train', train_images),
        #                   ('val', val_images), 
        #                   ('test', test_images)]:
        #     dest = os.path.join(output_dir, split, dataset, cls)
        #     os.makedirs(dest, exist_ok=True)
        #     for img in imgs:
        #         shutil.copy(os.path.join(src_path, img), 
        #                   os.path.join(dest, img))     

