import os, json, torch, argparse
from tqdm import tqdm
from glob import glob
from detector import GroundingDINOHandler

# def get_max_boxes(json_path):
#     with open(json_path, 'r', encoding='utf-8-sig') as f:
#         data = json.load(f)
#     return max(len(item.get("Position", [])) for item in data)

class BBoxProcessor:
    def __init__(self):
        self.args = self.parse_args()
        self.handler = self.init_handler()
        # Dynamically determine max_boxes from your annotation file
        # self.max_boxes = get_max_boxes(f"{self.args.root}/{self.args.output_json}")
        
    def parse_args(self):
        parser = argparse.ArgumentParser(description='Process images and save bounding box information')
        parser.add_argument('--config_path', type=str, default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", 
                          help='Path to GroundingDINO config file')
        # parser.add_argument('--weights_path', type=str, default="/home/bekhzod/Desktop/localization_models_performance/UzbekLicencePlateDetectorRecognizer/groundingdino_swint_ogc.pth",
        #                   help='Path to GroundingDINO weights file')
        parser.add_argument('--weights_path', type=str, default="groundingdino_swint_ogc.pth",
                          help='Path to GroundingDINO weights file')
        parser.add_argument('--device', type=str, default='cuda',
                          help='Device to use (cuda/cpu)')
        parser.add_argument('--root', type=str, required=True,
                          help='Root directory containing images')
        parser.add_argument('--output_json', type=str, required=True,
                          help='Output JSON file name')
        parser.add_argument('--caption', type=str, default='cells',
                          help='Caption text for detection')
        parser.add_argument('--box_threshold', type=float, default=0.05,
                          help='Box confidence threshold')
        parser.add_argument('--text_threshold', type=float, default=0.05,
                          help='Text confidence threshold')
        parser.add_argument('--iou_threshold', type=float, default=0.2,
                          help='IoU threshold for NMS')
        parser.add_argument('--max_box_size', type=float, default=0.8,
                          help='Max box size for NMS')
        parser.add_argument('--subfolder', type=str, default='normal',
                          help='Subfolder name for images')
        parser.add_argument('--image_ext', type=str, default='png',
                          help='Image file extension')
        return parser.parse_args()

    def init_handler(self):
        return GroundingDINOHandler(
            self.args.config_path,
            self.args.weights_path,
            device=self.args.device
        )

    def yolo_to_xyxy(self, bbox_norm: torch.Tensor, img_size: tuple) -> list:
        W, H = img_size
        cx = bbox_norm[0] * W
        cy = bbox_norm[1] * H
        w = bbox_norm[2] * W
        h = bbox_norm[3] * H

        return [
            int(cx - w/2),  # x_min
            int(cy - h/2),  # y_min
            int(cx + w/2),  # x_max
            int(cy + h/2)   # y_max
        ]

    def process_images(self):
        # im_paths = glob(f"{self.args.root}/*/{self.args.subfolder}/*.{self.args.image_ext}")
        im_paths = glob(f"{self.args.root}/{self.args.subfolder}/*/*/*.{self.args.image_ext}")

        result = []

        for idx, im_path in tqdm(enumerate(im_paths), desc="Processing images"):

            # if idx == 100: break
            
            base = os.path.basename(im_path)
            file_name = os.path.splitext(base)[0]
            image_pil, image_tensor = self.handler.load_image(im_path)
            W, H = image_pil.size

            boxes, phrases = self.handler.get_grounding_output(
                image_tensor, 
                caption=self.args.caption,
                box_threshold=self.args.box_threshold,
                text_threshold=self.args.text_threshold
            )
            
            boxes, phrases = self.handler.nms_by_confidence(
                boxes=boxes,
                phrases=phrases,
                iou_threshold=self.args.iou_threshold,  
                max_box_size=self.args.max_box_size
            )

            # # Convert all detected boxes
            # bboxes = []
            # for i in range(min(len(boxes), self.max_boxes)):
            #     bboxes.append(self.yolo_to_xyxy(boxes[i], (W, H)))
            # # Pad with [-1, -1, -1, -1] if needed
            # while len(bboxes) < self.max_boxes:
            #     bboxes.append([-1, -1, -1, -1])

            # result.append({
            #     "FileName": f"{file_name}_{self.args.subfolder}",
            #     "bboxes": bboxes
            # })

            bboxes = [-1, -1, -1, -1]
            # bboxes = [0, 0, 0, 0]
            if len(boxes) > 0:
                bboxes = self.yolo_to_xyxy(boxes[0], (W, H))

            result.append({
                "FileName": f"{file_name}_{self.args.subfolder}",
                "x": bboxes[0],
                "y": bboxes[1],
                "w": bboxes[2],
                "h": bboxes[3]
            })

        self.save_results(result)

    def save_results(self, new_data):

        json_path = f"{self.args.root}/{self.args.output_json}"
        
        if not os.path.exists(json_path):
            raise print(f"{json_path} file not found!")
        else:
            with open(json_path, 'r', encoding='utf-8-sig') as f:
                existing_data = json.load(f)        

        existing_filenames = {entry["FileName"] for entry in existing_data}
        for entry in new_data:
            if entry["FileName"] not in existing_filenames:
                existing_data.append(entry)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    processor = BBoxProcessor()
    processor.process_images()