import os, json, shutil, argparse, datetime
from glob import glob
from sklearn.model_selection import train_test_split

def load_data(json_path):
    """Load combined data with normal and abnormal cells"""
    with open(json_path) as f: 
        data = json.load(f)    
    return data

def create_directory_structure(root_dir, data_type, sample_type):
    """Create required directory hierarchy"""
    os.makedirs(os.path.join(root_dir, data_type, sample_type, 'annotations'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, data_type, sample_type, 'train'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, data_type, sample_type, 'val'), exist_ok=True)
    return {
        'annotations_dir': os.path.join(root_dir, data_type, sample_type, 'annotations'),
        'train_dir': os.path.join(root_dir, data_type, sample_type, 'train'),
        'val_dir': os.path.join(root_dir, data_type, sample_type, 'val')
    }

def generate_coco_annotations(data, img_paths, image_size=1024):
    """Generate full COCO-compatible annotations"""
    # Initialize COCO structure
    coco_data = {
        "info": {
            "description": "Dataset in COCO format",
            "version": "1.0",
            "year": datetime.datetime.now().year,
            "contributor": "Bekhzod Olimov",
            "date_created": datetime.datetime.now().strftime("%Y-%m-%d")
        },
        "licenses": [{
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",            
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"            
        }],
        "type": "instances",
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "normal", "supercategory": "none"},
            {"id": 1, "name": "abnormal", "supercategory": "none"}
        ]
    }
    
    # Initialize counters
    image_id, annotation_id = 1, 1     
    
    # Process each image
    for item in data:
        fname = item['FileName']
        if fname not in img_paths: continue
        # Add image information
        image_entry = {
            "id": image_id,
            "file_name": img_paths[fname],
            "width": image_size,
            "height": image_size,
            "date_captured": datetime.datetime.now().year
        }
        coco_data["images"].append(image_entry)
        
        # Process abnormal cells
        if 'Position' in item:
            for pos in item['Position']:
                # Convert to COCO bbox format [x, y, width, height]
                bbox = [pos['x'], pos['y'], pos['w'], pos['h']]
                area = pos['w'] * pos['h']
                segmentation = [
                    pos['x'], pos['y'],
                    pos['x'] + pos['w'], pos['y'],
                    pos['x'] + pos['w'], pos['y'] + pos['h'],
                    pos['x'], pos['y'] + pos['h']
                ]
                
                annotation_entry = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,  # abnormal
                    "bbox": bbox,
                    "area": area,
                    "segmentation": [segmentation],
                    "iscrowd": 0
                }
                coco_data["annotations"].append(annotation_entry)
                annotation_id += 1
        
        # Process normal cells
        if 'bboxes' in item:
            for bbox_coords in item['bboxes']:
                x_min, y_min, x_max, y_max = bbox_coords
                width = x_max - x_min
                height = y_max - y_min
                area = width * height
                segmentation = [
                    x_min, y_min,
                    x_max, y_min,
                    x_max, y_max,
                    x_min, y_max
                ]
                
                annotation_entry = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 0,  # normal
                    "bbox": [x_min, y_min, width, height],
                    "area": area,
                    "segmentation": [segmentation],
                    "iscrowd": 0
                }
                coco_data["annotations"].append(annotation_entry)
                annotation_id += 1
        
        image_id += 1
    
    return coco_data

def get_im_paths(json_path, sample_type):
    """Get image paths from directory structure"""
    all_img_paths = glob(f"{os.path.dirname(json_path)}/*/{sample_type}/*/*.png")
    # all_img_paths = glob(f"{os.path.dirname(json_path)}/*/*/*/*.png")
    img_paths = {}
    for im_path in all_img_paths:
        cls_name = os.path.dirname(im_path).split("/")[-3]
        fname    = os.path.basename(im_path) if cls_name == "abnormal" else f"{os.path.splitext(os.path.basename(im_path))[0]}_normal"
        img_paths[fname] = im_path
    return img_paths

def copy_images(data, img_paths, target_dir):
    """Copy images from source to target directory"""
    for item in data:        
        fname = item['FileName']        
        src_path = img_paths.get(fname)
        
        if not src_path:
            print(f"Warning: Image path not found for: {fname}")
            continue
            
        dst_path = os.path.join(target_dir, os.path.basename(src_path))
        
        if not os.path.exists(src_path):
            print(f"Warning: Image not found: {src_path}")
            continue
            
        shutil.copy2(src_path, dst_path)
        print(f"{src_path} copied to {dst_path}")

def main():
    parser = argparse.ArgumentParser(description='Prepare COCO dataset')
    parser.add_argument('--json_path', required=True, help='Path to combined JSON data')    
    parser.add_argument('--sample_type',  required=True, help='Sample type')
    parser.add_argument('--root', default='root', help='Root directory for dataset')    
    parser.add_argument('--image_size', type=int, default=1024, help='Image size (pixels)')
    parser.add_argument('--test_size', type=float, default=0.2, help='Validation split ratio')
    args = parser.parse_args()

    # Create directory structure
    data_type = os.path.dirname(args.json_path).split("/")[-1]
    dirs = create_directory_structure(args.root, data_type, args.sample_type)
    
    # Load data
    combined_data = load_data(args.json_path)

    # Get image paths
    img_paths = get_im_paths(json_path=args.json_path, sample_type=args.sample_type)    
    
    # Split data
    train_data, val_data = train_test_split(combined_data, test_size=args.test_size, random_state=42)    
    
    # Copy images to train/val directories
    copy_images(train_data, img_paths, dirs['train_dir']);  copy_images(val_data, img_paths, dirs['val_dir'])
    
    # Generate COCO annotations
    train_coco = generate_coco_annotations(train_data, img_paths, args.image_size)
    val_coco = generate_coco_annotations(val_data, img_paths, args.image_size)    
    
    # Save annotations
    with open(os.path.join(dirs['annotations_dir'], 'train_annotations.json'), 'w') as f: json.dump(train_coco, f, indent=4)
    with open(os.path.join(dirs['annotations_dir'], 'val_annotations.json'), 'w') as f: json.dump(val_coco, f, indent=4)
    
    print(f"Dataset preparation complete. Created:")
    print(f"- {len(train_data)} training samples")
    print(f"- {len(val_data)} validation samples")
    print(f"- {len(train_coco['annotations'])} training annotations")
    print(f"- {len(val_coco['annotations'])} validation annotations")
    print(f"Directory structure at: {args.root}")

if __name__ == "__main__":
    main()
