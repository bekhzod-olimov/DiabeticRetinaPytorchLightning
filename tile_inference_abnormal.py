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
from glob import glob
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
from data.roi_lbc import RoiLBC
from model import CellClassifier

class Inference:
    def __init__(self, im_dir, results_output_dir, device, save_model_path, save_data_path,
                 empty_bbox_type, slide_type, data_name, ckpt_path, save_wrong, to_catch,
                 run_name, tile_width=1024, tile_height=1024):

        self.im_dir     = im_dir
        self.slide_type = slide_type
        self.to_catch   = to_catch
        self.save_wrong = save_wrong
        self.run_name   = run_name
        self.data_name  = data_name
        self.ckpt_path  = ckpt_path
        self.save_model_path = save_model_path
        self.save_name = f"{run_name}_{slide_type.upper()}_{data_name}"
        self.results_output_dir = os.path.join(results_output_dir, f"{data_name}_{slide_type.upper()}", self.run_name)
        self.device = device
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.empty_bbox_type = empty_bbox_type      
    
        # with open(f"{save_data_path}/{self.save_name}_class_names.pkl", "rb") as fp: self.classes = pickle.load(fp)        
        # with open(f"{save_data_path}/{self.save_name}_class_counts.pkl", "rb") as fp: self.class_counts = pickle.load(fp)
        self.class_counts = {'normal': 95394, 'abnormal': 25095}
        self.classes = {'normal': 0, 'abnormal': 1}
        print(self.classes)        

    def get_tfs(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], im_size=1024): return T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])

    def load_models(self):

        print(f"ckpt_path -> {self.ckpt_path}")
        self.model = CellClassifier(run_name=self.run_name, empty_bbox_type=self.empty_bbox_type, class_counts=self.class_counts).to(self.device) 
        self.model.load_state_dict(torch.load(self.ckpt_path, weights_only=False)["state_dict"], strict=False)    
    
    def get_im(self, crop): return self.get_tfs()(crop).unsqueeze(dim=0)

    def process_tiff(self):
        self.load_models()
        os.makedirs(self.results_output_dir, exist_ok=True)        
             
        abnormal_dir = f"{self.results_output_dir}/{self.to_catch}"
        os.makedirs(abnormal_dir, exist_ok=True)       
        
        # im_paths = glob(f"{self.im_dir}/{self.slide_type.upper()}/*/*.png")
        
        im_paths = glob(f"{self.im_dir}/*.png")
        print(f"\nThere are {len(im_paths)} images found for inference!\n")
        tiles_count, normal_count = 0, 0
        
        for idx, im_path in tqdm(enumerate(im_paths), desc = "Inference..."):
            # if idx == 10: break
            tiles_count += 1
            fname = os.path.splitext(os.path.basename(im_path))[0]
            
            crop = Image.open(im_path).convert("RGB")            
            im = self.get_im(crop).to(self.device)
            with torch.no_grad(): pred = self.model(im)                
            pred_idx = torch.argmax(pred, dim = 1)                
            pred_prb = torch.nn.functional.softmax(pred, dim = 1)[0][pred_idx.item()]         
            class_name = list(self.classes.keys())[pred_idx.item()]            
            if class_name == self.to_catch:         
                normal_count += 1
                if self.save_wrong:
                    save_file_name = os.path.join(abnormal_dir, f"{fname}_{(pred_prb.item()*100):.2f}.jpg")
                    crop.save(save_file_name)
                                        
        print(f"Total number of tiles    -> {tiles_count}")
        print(f"Number of {self.to_catch} tiles -> {normal_count}")
        print(f"Accuracy score           -> {(1 - (normal_count / tiles_count)):.3f}")          

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--im_dir", required=True)  # directory containing TIFF images
    parser.add_argument('--run_name', type=str, required=True, help='Name of the training run')
    parser.add_argument('--empty_bbox_type', type=str, required=True, choices=['zeros', 'ones'], help='Empty bbox type')
    parser.add_argument("--results_output_dir", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_model_path", required=True)
    parser.add_argument("--save_data_path", required=True)
    parser.add_argument("--data_name", required=True)    
    parser.add_argument("--slide_type", required=True)    
    parser.add_argument("--to_catch", required=True)
    parser.add_argument('--save_wrong', action='store_true', help='Whether or not to save wrong predictions')
    parser.add_argument('--from_dir', action='store_true', help='Whether or not to save wrong predictions')

    args = parser.parse_args()    
    
    start_time = time.time()
    
    from glob import glob
    if args.from_dir:
        # print(f"args.data_name -> {ar gs.data_name}")        
        # print(f"args.slide_type -> {args.slide_type}")
        ckpt_paths = [ckpt for ckpt in glob(f"{os.path.dirname(args.ckpt_path)}/*.ckpt") if (args.data_name in ckpt) and (args.slide_type.upper() in ckpt)]
        # ckpt_paths = [ckpt for ckpt in glob(f"{os.path.dirname(args.ckpt_path)}/*.ckpt") if (args.data_name in ckpt) and ("PAP" in ckpt)]
        print(ckpt_paths)
        inf_start_time = time.time()
        for ckpt_path in ckpt_paths:                                     
            slide_dirs = glob(f"{args.im_dir}/{args.slide_type.upper()}/*")
            for slide_dir in slide_dirs:
                print(f"Prcocessing {os.path.basename(slide_dir)} slide")            
                slide_start_time = time.time()
                inf = Inference(
                    im_dir=slide_dir,
                    results_output_dir=args.results_output_dir,
                    device=args.device,
                    to_catch=args.to_catch,                
                    ckpt_path=ckpt_path,
                    empty_bbox_type=args.empty_bbox_type,
                    save_wrong=args.save_wrong,
                    save_model_path=args.save_model_path,
                    save_data_path=args.save_data_path,
                    data_name=args.data_name,
                    run_name=args.run_name,
                    slide_type=args.slide_type                
                )
                inf.process_tiff()
                print(f"\nSlide inference completed in {(time.time() - slide_start_time):.1f} seconds.\n")  
    else:
        print(f"device -> {args.device}")
        inf_start_time = time.time()
        slide_dirs = glob(f"{args.im_dir}/{args.slide_type.upper()}/*")
        for slide_dir in slide_dirs:
            slide_start_time = time.time()
            print(f"Prcocessing {os.path.basename(slide_dir)} slide")
            inf = Inference(
                im_dir=slide_dir,
                results_output_dir=args.results_output_dir,
                device=args.device,
                ckpt_path=args.ckpt_path,
                to_catch=args.to_catch,                
                empty_bbox_type=args.empty_bbox_type,
                save_wrong=args.save_wrong,
                save_model_path=args.save_model_path,
                save_data_path=args.save_data_path,
                data_name=args.data_name,
                run_name=args.run_name,
                slide_type=args.slide_type                
            )
            inf.process_tiff()
            print(f"\nSlide inference completed in {(time.time() - slide_start_time):.1f} seconds.\n")  

    print(f"\nInference completed in {(time.time() - inf_start_time):.1f} seconds.")  