import os, sys, torch, numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms
sys.path.append("./GroundingDINO")
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
import groundingdino.datasets.transforms as T

class GroundingDINOHandler:
    def __init__(self, model_config_path, model_checkpoint_path, device):
        self.device = device
        self.model = self.load_model(model_config_path, model_checkpoint_path)
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def load_model(self, model_config_path, model_checkpoint_path):
        args = SLConfig.fromfile(model_config_path)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        return model.to(self.device)

    def load_image(self, image_path):
        image_pil = Image.open(image_path).convert("RGB")
        image, _ = self.transform(image_pil, None)
        return image_pil, image

    def plot_boxes_to_image(self, image_pil, tgt, lbl_path):
        H, W = tgt["size"]
        boxes = tgt["boxes"]
        labels = tgt["labels"]        
        draw = ImageDraw.Draw(image_pil)
        mask = Image.new("L", image_pil.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        font = ImageFont.load_default()

        for box, label in zip(boxes, labels):
            box = box.to("cpu")
            box = box * torch.Tensor([W, H, W, H])            
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            x0, y0, x1, y1 = box.int().tolist()
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
            label_text = str(label)
            bbox = draw.textbbox((x0, y0), label_text, font) if hasattr(draw, "textbbox") else (x0, y0, x0 + 50, y0 + 10)
            draw.rectangle(bbox, fill=color)
            draw.text((x0, y0), label_text, fill="white", font=font)
            mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cx, cy, w, h = map(float, parts[-4:])
                class_name = " ".join(parts[:-4])                

                if class_name in ["normal", "abnormal cell"]: continue                
                
                x0 = int((cx - w / 2) * W)
                y0 = int((cy - h / 2) * H)
                x1 = int((cx + w / 2) * W)
                y1 = int((cy + h / 2) * H)
                draw.rectangle([x0, y0, x1, y1], outline='red', width=3)
                text = f"GT: {class_name}"
                bbox = draw.textbbox((x0, y0), text, font) if hasattr(draw, "textbbox") else (x0, y0, x0 + 50, y0 + 10)
                draw.rectangle(bbox, fill='red')
                draw.text((x0, y0), text, fill="white", font=font)

        return image_pil, mask        

    def get_grounding_output(self, image, caption, box_threshold, text_threshold=None, with_logits=True, token_spans=None):    
        
        assert text_threshold is not None or token_spans is not None
        caption = caption.lower().strip()
        caption = caption if caption.endswith(".") else caption + "."
        image = image.to(self.device)
        with torch.no_grad(): outputs = self.model(image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]
        boxes = outputs["pred_boxes"][0]

        if token_spans is None:
            filt_mask = logits.max(dim=1)[0] > box_threshold
            logits_filt = logits[filt_mask]
            boxes_filt = boxes[filt_mask]
            tokenized = self.model.tokenizer(caption)
            pred_phrases = [get_phrases_from_posmap(logit > text_threshold, tokenized, self.model.tokenizer) +
                            (f"({str(logit.max().item())[:4]})" if with_logits else "")
                            for logit in logits_filt]
        else:
            positive_maps = create_positive_map_from_span( self.model.tokenizer(caption), token_span=token_spans ).to(image.device)
            logits_for_phrases = positive_maps @ logits.T
            all_boxes, all_phrases = [], []
            for token_span, logit_phr in zip(token_spans, logits_for_phrases):
                phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
                filt_mask = logit_phr > box_threshold
                all_boxes.append(boxes[filt_mask])
                if with_logits:
                    all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr[filt_mask]])
                else:
                    all_phrases.extend([phrase] * filt_mask.sum().item())
            boxes_filt = torch.cat(all_boxes, dim=0).cpu()
            pred_phrases = all_phrases

        return boxes_filt, pred_phrases

    @staticmethod
    def xywh_to_xyxy(boxes):
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)

    def extract_confidences(self, pred_phrases): return torch.tensor([float(p.split('(')[-1].rstrip(')')) if '(' in p else 1.0 for p in pred_phrases], device=self.device)

    def nms_by_confidence(self, boxes, phrases, iou_threshold=0.2, max_box_size=0.8):
        widths = boxes[:, 2]
        heights = boxes[:, 3]
        size_mask = (widths <= max_box_size) & (heights <= max_box_size)
        boxes = boxes[size_mask]
        phrases = [phrases[i] for i in torch.where(size_mask)[0]]

        if len(boxes) == 0: return boxes, []

        confidences = self.extract_confidences(phrases)
        boxes_xyxy = self.xywh_to_xyxy(boxes)        
        keep = nms(boxes_xyxy, confidences, iou_threshold)
        return boxes[keep], [phrases[i] for i in keep]

    @staticmethod
    def filter_large_bboxes_with_phrases(boxes, phrases, max_area=0.3):
        areas = boxes[:, 2] * boxes[:, 3]
        mask = areas < max_area
        filtered_boxes = boxes[mask]
        filtered_phrases = [p for i, p in enumerate(phrases) if mask[i]]
        return filtered_boxes, filtered_phrases

    @staticmethod
    def compute_iou(box1, box2):
        x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter_area
        return inter_area / union if union > 0 else 0