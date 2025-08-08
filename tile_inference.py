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
    def __init__(self, tiff_path, results_output_dir, device, save_model_path, save_data_path,
                 empty_bbox_type, slide_type, data_name, ckpt_path, save_wrong, 
                 run_name, tile_width=1024, tile_height=1024):

        self.tiff_path  = tiff_path
        self.slide_type = slide_type
        self.run_name   = run_name
        self.data_name  = data_name
        self.ckpt_path = ckpt_path
        self.save_model_path = save_model_path
        self.save_name = f"{run_name}_{slide_type.upper()}_{data_name}"
        self.results_output_dir = f"{results_output_dir}/{data_name}_{slide_type.upper()}"        
        self.device = device
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.empty_bbox_type = empty_bbox_type 
        self.save_wrong = save_wrong
        
        # with open(f"{save_data_path}/{self.save_name}_class_names.pkl", "rb") as fp: self.classes = pickle.load(fp)
        # with open(f"{save_data_path}/{self.save_name}_class_counts.pkl", "rb") as fp: self.class_counts = pickle.load(fp)
        self.class_counts = {'normal': 95394, 'abnormal': 25095}
        self.classes = {'normal': 0, 'abnormal': 1}
        print(self.classes)        

    def get_tfs(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], im_size=1024):
        return T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])

    def load_models(self):

        print(f"ckpt_path -> {self.ckpt_path}")           
        self.model = CellClassifier(run_name=self.run_name, empty_bbox_type=self.empty_bbox_type, class_counts=self.class_counts).to(self.device) 
        self.model.load_state_dict(torch.load(self.ckpt_path, weights_only=False)["state_dict"], strict=False)

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
        self.load_models()
        os.makedirs(self.results_output_dir, exist_ok=True)

        if not os.path.isfile(self.tiff_path): print("File is not found"); return
        
        slide = tiff.imread(self.tiff_path)
        
        # Handle grayscale or multi-channel images
        if len(slide.shape) == 2: slide = np.stack([slide] * 3, axis=-1)  # Convert grayscale to RGB
        
        if self.slide_type == "lbc":

            x_min, y_min, x_max, y_max = self.get_roi()
            slide = slide[y_min:y_max, x_min:x_max]

        fname = os.path.splitext(os.path.basename(self.tiff_path))[0]        
        
        im = cv2.resize(slide, dsize=(1024,1024))
        cv2.imwrite(filename=f"{self.results_output_dir}/{fname}_ROI.jpg", img=im)
        
        height, width = slide.shape[:2]
        
        save_folder = os.path.join(self.results_output_dir, fname)
        # os.makedirs(save_folder, exist_ok=True)        
        abnormal_dir = f"{save_folder}_crop/abnormal" if self.slide_type == "lbc" else f"{save_folder}/abnormal"
        os.makedirs(abnormal_dir, exist_ok=True)       
        
        tiles_count, abnormal_count = 0, 0            

        for x in tqdm(range(0, width, self.tile_width), desc="Processing..."):
            # if x == 2 * self.tile_width: break
            for y in range(0, height, self.tile_height):
                tiles_count += 1
                # if y == self.tile_height: break
                crop_width = min(self.tile_width, width - x)
                crop_height = min(self.tile_height, height - y)
                crop = slide[y:y + crop_height, x:x + crop_width]

                crop = self.bgr2rgb(crop)
                im = self.get_im(crop).to(self.device)
                with torch.no_grad(): pred = self.model(im)                
                pred_idx = torch.argmax(pred, dim = 1)                
                pred_prb = torch.nn.functional.softmax(pred, dim = 1)[0][pred_idx.item()]                
                class_name = list(self.classes.keys())[pred_idx.item()]
                if class_name == "abnormal":
                    abnormal_count += 1   
                    if self.save_wrong:
                        save_file_name = os.path.join(abnormal_dir, f"{fname}_{x}_{y}_{(pred_prb.item()*100):.2f}.jpg")
                        crop.save(save_file_name)
                                                     
        print(f"Total number of tiles    -> {tiles_count}")
        print(f"Number of abnormal tiles -> {abnormal_count}")
        print(f"Accuracy score           -> {(1 - (abnormal_count / tiles_count)):.3f}")                    

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--tiff_path", required=True)
#     parser.add_argument("--results_output_dir", required=True)
#     parser.add_argument("--device", default="cpu")
#     parser.add_argument("--save_model_path", required=True)
#     parser.add_argument("--save_data_path", required=True)
#     parser.add_argument("--data_name", required=True)
#     parser.add_argument("--run_name", required=True)
#     parser.add_argument("--slide_type", required=True)
#     parser.add_argument("--project_type", required=True) 

#     args = parser.parse_args()

#     start_time = time.time()
#     inf = Inference(
#         tiff_path=args.tiff_path,
#         results_output_dir=args.results_output_dir,
#         device=args.device,
#         save_model_path=args.save_model_path,
#         save_data_path=args.save_data_path,
#         data_name=args.data_name,
#         run_name=args.run_name,
#         slide_type=args.slide_type,
#         project_type=args.project_type        
#     )
#     inf.process_tiff()
#     print(f"Inference is completed in {(time.time() - start_time):.3f} secs!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiff_dir", required=True)  # directory containing TIFF images
    parser.add_argument('--run_name', type=str, required=True, help='Name of the training run')
    parser.add_argument('--empty_bbox_type', type=str, required=True, choices=['zeros', 'ones'], help='Empty bbox type')
    parser.add_argument("--results_output_dir", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_model_path", required=True)
    parser.add_argument("--save_data_path", required=True)
    parser.add_argument("--data_name", required=True)    
    parser.add_argument("--slide_type", required=True)    
    parser.add_argument('--save_wrong', action='store_true', help='Whether or not to save wrong predictions')
    parser.add_argument('--from_dir', action='store_true', help='Whether or not to save wrong predictions')

    args = parser.parse_args()

    tiff_files = [f for f in os.listdir(args.tiff_dir) if f.endswith(".tiff") or f.endswith(".tif")]
    
    start_time = time.time()        

    if args.from_dir:
        # print(f"args.data_name -> {args.data_name}")
        # print(f"args.slide_type -> {args.slide_type}")
        from glob import glob
        print(glob(f"{os.path.dirname(args.ckpt_path)}/*.ckpt"))
        ckpt_paths = [ckpt for ckpt in glob(f"{os.path.dirname(args.ckpt_path)}/*.ckpt") if (args.data_name in ckpt) and (args.slide_type.upper() in ckpt)]
        print(ckpt_paths)
        for ckpt_path in ckpt_paths:    
            print(f"Using {ckpt_path}")            

            print(f"device -> {args.device}")
            inf_start_time = time.time()
            for i, tiff_filename in enumerate(tiff_files):
                print(f"\nProcessing ({i+1}/{len(tiff_files)}): {tiff_filename}")
                tiff_path = os.path.join(args.tiff_dir, tiff_filename)

                print(f"device -> {args.device}")
                inf_start_time = time.time()
                inf = Inference(
                    tiff_path=tiff_path,
                    results_output_dir=args.results_output_dir,
                    device=args.device,
                    # ckpt_path=args.ckpt_path,
                    save_wrong=args.save_wrong,
                    ckpt_path=ckpt_path,
                    empty_bbox_type=args.empty_bbox_type,
                    save_model_path=args.save_model_path,
                    save_data_path=args.save_data_path,
                    data_name=args.data_name,
                    run_name=args.run_name,
                    slide_type=args.slide_type                
                )
                inf.process_tiff()
                print(f"\nInference completed in {(time.time() - inf_start_time):.1f} seconds.")                
        
                del inf
                gc.collect()
    else:
        print(f"device -> {args.device}")
        for i, tiff_filename in enumerate(tiff_files):
            print(f"\nProcessing ({i+1}/{len(tiff_files)}): {tiff_filename}")
            tiff_path = os.path.join(args.tiff_dir, tiff_filename)

            print(f"device -> {args.device}")
            inf_start_time = time.time()
            inf = Inference(
                tiff_path=tiff_path,
                results_output_dir=args.results_output_dir,
                device=args.device,
                ckpt_path=args.ckpt_path,
                save_wrong=args.save_wrong,
                # ckpt_path=ckpt_path,
                empty_bbox_type=args.empty_bbox_type,
                save_model_path=args.save_model_path,
                save_data_path=args.save_data_path,
                data_name=args.data_name,
                run_name=args.run_name,
                slide_type=args.slide_type                
            )
            inf.process_tiff()
            print(f"\nInference completed in {(time.time() - inf_start_time):.1f} seconds.")                
        
            del inf
            gc.collect()

    # for i, tiff_filename in enumerate(tiff_files):
    #     print(f"\nProcessing ({i+1}/{len(tiff_files)}): {tiff_filename}")
    #     tiff_path = os.path.join(args.tiff_dir, tiff_filename)

    #     print(f"device -> {args.device}")
    #     inf_start_time = time.time()
    #     inf = Inference(
    #         tiff_path=tiff_path,
    #         results_output_dir=args.results_output_dir,
    #         device=args.device,
    #         ckpt_path=args.ckpt_path,
    #         save_wrong=args.save_wrong,
    #         # ckpt_path=ckpt_path,
    #         empty_bbox_type=args.empty_bbox_type,
    #         save_model_path=args.save_model_path,
    #         save_data_path=args.save_data_path,
    #         data_name=args.data_name,
    #         run_name=args.run_name,
    #         slide_type=args.slide_type                
    #     )
    #     inf.process_tiff()
    #     print(f"\nInference completed in {(time.time() - inf_start_time):.1f} seconds.")                
        
    #     del inf
    #     gc.collect()

    print(f"\nAll inferences completed in {(time.time() - start_time):.1f} seconds.")