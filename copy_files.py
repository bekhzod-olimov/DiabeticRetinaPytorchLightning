import os
import shutil
import argparse

class ImageFolderCopier:
    def __init__(self, root_dir, save_dir):
        self.root_dir = root_dir
        self.save_dir = save_dir
        self.categories = ['LBC', 'PAP']
        self.subcategories = ['normal', 'abnormal']

    def create_target_dirs(self):
        for category in self.categories:
            for subcategory in self.subcategories:
                target_dir = os.path.join(self.save_dir, category, subcategory)
                os.makedirs(target_dir, exist_ok=True)

    def copy_images(self):
        for subcategory in self.subcategories:
            subcategory_path = os.path.join(self.root_dir, subcategory)
            if not os.path.isdir(subcategory_path):
                continue
            for category in self.categories:
                category_path = os.path.join(subcategory_path, category)
                if not os.path.isdir(category_path):
                    continue
                # Traverse all subsubsubdirs under category_path
                for current_root, dirs, files in os.walk(category_path):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                            src_file = os.path.join(current_root, file)
                            # To avoid overwriting, prefix with subdir structure
                            rel_path = os.path.relpath(current_root, category_path)
                            prefix = rel_path.replace(os.sep, '_')
                            if prefix == '.' or prefix == '':
                                dst_filename = file
                            else:
                                dst_filename = f"{prefix}_{file}"
                            dst_file = os.path.join(self.save_dir, category, subcategory, dst_filename)
                            shutil.copy2(src_file, dst_file)
        print(f"Copying completed from {self.root_dir} to {self.save_dir} with the specified structure.")

    @classmethod
    def from_argparse(cls):
        parser = argparse.ArgumentParser(description="Copy images from root to save_dir with specific structure.")
        # parser.add_argument('--root_dir', type=str, default="/mnt/data/cervical_screening/classification/hospital_dataset/classification/tile_classification/100", help='Root directory to copy from')
        # parser.add_argument('--save_dir', type=str, default="/mnt/data/cervical_screening/classification/hospital_dataset/classification/tile_classification/test_copying", help='Directory to copy images to')

        parser.add_argument('--root_dir', type=str, default="/vol0/nfs9/tileimage/new/1024", help='Root directory to copy from')        
        parser.add_argument('--save_dir', type=str, default="/vol0/nfs9/tileimage/new/1024/25_05_16", help='Directory to copy images to')

        
        args = parser.parse_args()
        return cls(args.root_dir, args.save_dir)

    def run(self):
        self.create_target_dirs()
        self.copy_images()

if __name__ == "__main__":
    copier = ImageFolderCopier.from_argparse()
    copier.run()
