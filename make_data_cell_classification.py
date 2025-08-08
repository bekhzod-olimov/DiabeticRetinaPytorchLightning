import os
import json
from PIL import Image

def crop_cells_from_coco_annotations(images_dir, annotations_path, output_dir):
    # Load COCO annotations
    with open(annotations_path, 'r') as f:
        coco = json.load(f)

    # Create output directories for normal and abnormal cells
    normal_dir = os.path.join(output_dir, 'normal')
    abnormal_dir = os.path.join(output_dir, 'abnormal')
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(abnormal_dir, exist_ok=True)

    # Map image_id to file_name
    image_id_to_file = {img['id']: img['file_name'] for img in coco['images']}
    # Map category_id to category name
    category_id_to_name = {cat['id']: cat['name'] for cat in coco['categories']}

    # Group annotations by image_id
    from collections import defaultdict
    annotations_by_image = defaultdict(list)
    for ann in coco['annotations']:
        annotations_by_image[ann['image_id']].append(ann)

    # Process each image and crop cells
    for image_id, anns in annotations_by_image.items():
        image_path = image_id_to_file[image_id]
        full_image_path = os.path.join(images_dir, os.path.basename(image_path))

        if not os.path.exists(full_image_path):
            print(f"Warning: Image file not found: {full_image_path}")
            continue

        with Image.open(full_image_path) as img:
            img_w, img_h = img.size
            for idx, ann in enumerate(anns):
                category_name = category_id_to_name[ann['category_id']]
                x, y, w, h = map(int, ann['bbox'])
                # Clamp coordinates to image boundaries
                left = max(0, x)
                upper = max(0, y)
                right = min(img_w, x + w)
                lower = min(img_h, y + h)
                # Only crop if box is valid
                if right > left and lower > upper:
                    cropped_cell = img.crop((left, upper, right, lower))
                    save_dir = normal_dir if category_name == 'normal' else abnormal_dir
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    cell_filename = f"{base_name}_cell{idx+1}_{category_name}.png"
                    save_path = os.path.join(save_dir, cell_filename)
                    cropped_cell.save(save_path)
                else:
                    print(f"Skipped invalid crop for {full_image_path}, annotation {ann['id']}")

    print(f"Cropping complete. Cells saved in {normal_dir} and {abnormal_dir}")

# Example usage:
crop_cells_from_coco_annotations(
    images_dir='/vol0/nfs9/tileimage/new/detection/25_06_09/LBC/train',
    annotations_path='/vol0/nfs9/tileimage/new/detection/25_06_09/LBC/annotations/train_annotations.json',
    output_dir='/vol0/nfs9/tileimage/new/cell_classification/25_06_09/LBC/train'
)
