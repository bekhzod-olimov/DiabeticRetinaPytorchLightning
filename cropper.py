import os
import gc
import torch
import tifffile as tiff
import timm
import pickle
import cv2
import time
import numpy as np
import argparse
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
from data.roi_lbc import RoiLBC
from model import CellClassifier

class Inference:
    def __init__(self, tiff_path, results_output_dir, device, slide_type, data_name, run_name, tile_width=1024, tile_height=1024):

        self.tiff_path  = tiff_path
        self.slide_type = slide_type
        self.run_name   = run_name                  
        self.results_output_dir = os.path.join(results_output_dir, data_name, slide_type.upper())      
        self.device = device
        self.tile_width = tile_width
        self.tile_height = tile_height          
        

    def get_tfs(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], im_size=1024): return T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    

    def get_roi(self):
        
        tile, size_0, size_8 = self.read_slide()
        roi = RoiLBC()
        bbox = roi.main_method(tile)
        print(f"Bounding Box: {bbox}")
        
        original_width = size_0[0]
        original_height = size_0[1]
        resized_width = size_8[0]
        resized_height = size_8[1]
        x_ratio = original_width / resized_width
        y_ratio = original_height / resized_height
        
        bbox = [int(bbox[0] * x_ratio), int(bbox[1] * y_ratio), int((bbox[0]+bbox[2]) * x_ratio), int((bbox[1]+(bbox[3])) * y_ratio)]

        print(f"Original Bounding Box: {bbox}")

        return bbox   
    
    def get_im(self, crop): return self.get_tfs()(crop).unsqueeze(dim=0)

    def read_slide(self):
        
        # tiff path file
        with tiff.TiffFile(self.tiff_path) as slide:

            # slid 2 size
            size_0 = slide.pages[0].shape
            print(f"Original slide size: {size_0}")
            size_8 = slide.pages[8].shape
            print(f"8th layer slide size: {size_8}")            
        
            tile = slide.pages[8].asarray()
            tile = Image.fromarray(tile)
            tile = tile.convert("RGB")           

            original_image = tile.copy()         

        return original_image, size_0, size_8
    
    def bgr2rgb(self, image_array):
        image = Image.fromarray(image_array.astype(np.uint8))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image

    def process_tiff(self):        
        os.makedirs(self.results_output_dir, exist_ok=True)

        if not os.path.isfile(self.tiff_path): print("File is not found"); return
        
        slide = tiff.imread(self.tiff_path)
        
        # Handle grayscale or multi-channel images
        if len(slide.shape) == 2: slide = np.stack([slide] * 3, axis=-1)  # Convert grayscale to RGB
        
        if self.slide_type == "lbc":

            x_min, y_min, x_max, y_max = self.get_roi()
            slide = slide[y_min:y_max, x_min:x_max]

        fname = os.path.splitext(os.path.basename(self.tiff_path))[0] 
        os.makedirs(os.path.join(self.results_output_dir, fname), exist_ok=True)
               
        
        im = cv2.resize(slide, dsize=(1024,1024))
        cv2.imwrite(filename=f"{self.results_output_dir}/{fname}_ROI.jpg", img=im)
        
        height, width = slide.shape[:2]                        

        for x in tqdm(range(0, width, self.tile_width), desc="Processing..."):
            # if x == self.tile_width: break
            for y in range(0, height, self.tile_height):
                # if y == self.tile_height: break                          
                crop_width = min(self.tile_width, width - x)
                crop_height = min(self.tile_height, height - y)
                crop = slide[y:y + crop_height, x:x + crop_width]
                crop = self.bgr2rgb(crop)
                save_file_name = os.path.join(self.results_output_dir, fname, f"{x}_{y}.png")
                crop.save(save_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiff_dir", required=True)  # directory containing TIFF images
    parser.add_argument('--run_name', type=str, required=True, help='Name of the training run')    
    parser.add_argument('--data_name', type=str, required=True, help='Name of the training run')    
    parser.add_argument("--results_output_dir", required=True)    
    parser.add_argument("--device", default="cuda")            
    parser.add_argument("--slide_type", required=True)    

    args = parser.parse_args()

    tiff_files = [f for f in os.listdir(args.tiff_dir) if f.endswith(".tiff") or f.endswith(".tif")]
    
    start_time = time.time()    

    for i, tiff_filename in enumerate(tiff_files):
        print(f"\nProcessing ({i+1}/{len(tiff_files)}): {tiff_filename}")
        tiff_path = os.path.join(args.tiff_dir, tiff_filename)

        print(f"device -> {args.device}")
        inf_start_time = time.time()
        inf = Inference(
            tiff_path=tiff_path,
            results_output_dir=args.results_output_dir,
            device=args.device, 
            data_name=args.data_name,           
            run_name=args.run_name,
            slide_type=args.slide_type                
        )
        inf.process_tiff()
        print(f"\nInference completed in {(time.time() - inf_start_time):.1f} seconds.")                
        
        del inf
        gc.collect()

    print(f"\nAll inferences completed in {(time.time() - start_time):.1f} seconds.")