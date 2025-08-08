import os
import gc
import torch
import onnxruntime as ort
# import tifffile as tiff
from time import time
import numpy as np
import argparse
from torchvision import transforms as TS
from torchvision.ops import nms
from PIL import Image
from tqdm import tqdm
from data.roi_lbc import RoiLBC
from PIL import Image, ImageDraw, ImageFont
from glob import glob
from model import CellClassifier
from transformations import get_fts

import sys

def resize_with_aspect_ratio(image, size, interpolation=Image.BILINEAR):
    """Resizes an image while maintaining aspect ratio and pads it."""
    original_width, original_height = image.size
    ratio = min(size / original_width, size / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    
    image = image.resize((new_width, new_height), interpolation)

    # Create a new image with the desired size and paste the resized image onto it
    new_image = Image.new("RGB", (size, size))
    new_image.paste(image, ((size - new_width) // 2, (size - new_height) // 2))
    return new_image, ratio, (size - new_width) // 2, (size - new_height) // 2

def iou(box1, box2):
    # box: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def remove_overlapping_boxes(boxes, scores, labels, overlap_thresh=None):
    if len(boxes) == 0:
        return boxes, scores, labels

    boxes = np.array(boxes)
    scores = np.array(scores)
    labels = np.array(labels)
    idxs = np.argsort(scores)[::-1]
    keep = []

    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)
        rest = idxs[1:]
        filtered_idxs = []
        for i in rest:
            if iou(boxes[current], boxes[i]) <= overlap_thresh:
                filtered_idxs.append(i)
        idxs = np.array(filtered_idxs)
    keep = np.array(keep)
    return boxes[keep], scores[keep], labels[keep]

def draw(images, labels, boxes, scores, ratios, paddings, cls_names=["normal", "abnormal"], thrh=0.2, overlap_thresh=0.3):    
    result_images = []
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scr = scr[scr > thrh]

        ratio = ratios[i]
        pad_w, pad_h = paddings[i]

        # Adjust bounding boxes according to the resizing and padding
        adjusted_boxes = [
            [
                (bb[0] - pad_w) / ratio,
                (bb[1] - pad_h) / ratio,
                (bb[2] - pad_w) / ratio,
                (bb[3] - pad_h) / ratio,
            ] for bb in box
        ]

        # Remove overlapping boxes        
        # print(len(adjusted_boxes))
        filtered_boxes, filtered_scores, filtered_labels = remove_overlapping_boxes(
            adjusted_boxes, scr, lab, overlap_thresh=overlap_thresh
        )

        # print(len(filtered_boxes))

        for lbl, bb, ss in zip(filtered_labels, filtered_boxes, filtered_scores):
            lbl_name = cls_names[lbl]
            bbox_color = "red" if lbl_name == "abnormal" else "green"            
            draw.rectangle(bb.tolist(), outline=bbox_color, width=3)
            draw.text((bb[0], bb[1]), text=f"{str(lbl_name)} {ss:.2f}", fill="blue", font=font)
        result_images.append(im)
    return result_images


def process_image(sess, im_pil, input_path, save_dir):
    start_time = time()
    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.splitext(os.path.basename(input_path))[0]
    # Resize image while preserving aspect ratio
    resized_im_pil, ratio, pad_w, pad_h = resize_with_aspect_ratio(im_pil, 640)
    orig_size = torch.tensor([[resized_im_pil.size[1], resized_im_pil.size[0]]])

    transforms = TS.Compose(
        [
            TS.ToTensor(),
        ]
    )
    im_data = transforms(resized_im_pil).unsqueeze(0)    

    output = sess.run(
        output_names=None,
        input_feed={"images": im_data.numpy(), "orig_target_sizes": orig_size.numpy()},
    )

    labels, boxes, scores = output    
    result_images = draw([im_pil], labels, boxes, scores, [ratio], [(pad_w, pad_h)])
    result_images[0].save(f"{save_dir}/{fname}_result.jpg")
    print(f"Image processing is completed in {(time() - start_time):.10f} secs. Result saved in {save_dir} folder.")    

class Inference:
    def __init__(self, tiff_path, sess, results_output_dir, device, ckpt_path,
                 slide_type, run_name, tile_width=1024, tile_height=1024):

        self.tiff_path  = tiff_path
        self.slide_type = slide_type   
        self.sess = sess     
        self.results_output_dir = f"{results_output_dir}/{slide_type.upper()}"        
        self.device = device
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.run_name = run_name     
        self.ckpt_path = ckpt_path                
        self.classes = {"normal": 0, "abnormal": 1}

    # def get_tfs(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], im_size=1024): return TS.Compose([TS.Resize((im_size, im_size)), TS.ToTensor(), TS.Normalize(mean=mean, std=std)])
    def apply_tfs(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]): return get_fts(mean=mean, std=std)[1]

    def load_models(self):

        start = time()

        print(f"ckpt_path -> {self.ckpt_path}")
        self.model = CellClassifier(run_name=self.run_name, empty_bbox_type="ones", class_counts={'normal': 95394, 'abnormal': 25095}).to(self.device) 
        self.model.load_state_dict(torch.load(self.ckpt_path, weights_only=False)["state_dict"], strict=False)
        print(f"\nTile classification models are loaded in {time() - start} secs!\n")      
    
    def get_roi(self):
        
        start = time()
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
        print(f"\nROI is obtained in {time() - start} secs!\n")

        return bbox   
    
    def get_im(self, crop): return self.apply_tfs()(crop).unsqueeze(dim=0)

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
        
        read_slide_start = time()
        im_paths = glob(f"{self.tiff_path}/*.png")     

        fname = os.path.basename(self.tiff_path)       
        
        save_folder = os.path.join(self.results_output_dir, fname)        
        abnormal_dir = f"{save_folder}/abnormal"
        normal_dir = f"{save_folder}/normal"
        os.makedirs(abnormal_dir, exist_ok=True);os.makedirs(normal_dir, exist_ok=True)
        
        tiles_count, abnormal_count = 0, 0                    

        for im_path in tqdm(im_paths, desc = f"Processing {fname} dir..."):
            tiles_count += 1                       

            crop = Image.open(im_path).convert("RGB")            
            im = self.get_im(crop).to(self.device)  
            with torch.no_grad(): pred = self.model(im)
            ensemble_pred = torch.argmax(pred, dim=1).item()
            class_name = list(self.classes.keys())[ensemble_pred]
            if class_name == "normal":                
                save_file_name = os.path.join(normal_dir, f"{fname}_{tiles_count}.png")
                crop.save(save_file_name)
            
            elif class_name == "abnormal":
                cell_start = time()                

                
                # Load the ONNX model
                # sess = ort.InferenceSession(args.onnx)
                # print(f"Using device: {ort.get_device()}")                
                
                im_pil = Image.open(im_path).convert("RGB")
                process_image(self.sess, im_pil, im_path, abnormal_dir)                                

                # print(f"\n{args.n_ims} 이미지와 추론 시간은 {(completed_time):.5f} 초입니다!\n")
                # print(f"이미지당 평균 시간은 {(completed_time / args.n_ims):.5f} 초입니다!\n")

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--tiff_dir", required=True)  # directory containing TIFF images
    parser.add_argument("--tiff_dir", required=True)  # directory containing TIFF images
    parser.add_argument("--results_output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ckpt_path", required=True)        
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--slide_type", required=True)    

    args = parser.parse_args()

    start_time = time()
    # tiff_files = [f for f in os.listdir(args.tiff_dir) if f.endswith(".tiff") or f.endswith(".tif")]

    # for i, tiff_filename in enumerate(tiff_files):
    #     print(f"\nProcessing ({i+1}/{len(tiff_files)}): {tiff_filename}")
    #     tiff_path = os.path.join(args.tiff_dir, tiff_filename)
        
    providers = [
                ('CUDAExecutionProvider', {'device_id': 0}),  # 0 is the default GPU
                'CPUExecutionProvider'
                ]    

    sess = ort.InferenceSession("/home/bekhzod/Desktop/backup/dfine/outputs/dfine_hgnetv2_l_obj2custom_100_epochs/best_stg1.onnx", providers=providers)
    print(f"\n\n\nUsing device: {sess.get_providers()}\n\n\n")
    
    slide_dirs = glob(f"{args.tiff_dir}/{args.slide_type.upper()}/*")
    print(slide_dirs)
    for slide_dir in slide_dirs:
        slide_start_time = time()
        print(f"Prcocessing {os.path.basename(slide_dir)} slide")    
        print(f"device -> {args.device}")
        inf_start_time = time()
        inf = Inference(
            tiff_path=slide_dir,
            sess = sess,
            results_output_dir=args.results_output_dir,
            device=args.device,
            ckpt_path=args.ckpt_path,            
            run_name=args.run_name,
            slide_type=args.slide_type,            
        )
        inf.process_tiff()
        print(f"\nSlide inference completed in {(time() - slide_start_time):.1f} seconds.\n")          
        
        # # ✅ Delete object and run garbage collector to free RAM
        # del inf
        # gc.collect()

    print(f"\nAll inferences completed in {(time() - start_time):.1f} seconds.")