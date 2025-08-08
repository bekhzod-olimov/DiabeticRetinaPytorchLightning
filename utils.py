import os, json, torch, random
import numpy as np
from sklearn.model_selection import train_test_split
from glob import glob; from torch import nn

def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# def get_class_name(path): return os.path.dirname(path).split("/")[-1]
def get_class_name(path): return os.path.dirname(path).split("/")[-3]

def get_labels(paths): return [get_class_name(path) for path in paths]

def get_splits(root, sample_type, random_state):
    
    # all_image_paths = glob(f"{root}/{sample_type}/*/*.png")
    all_image_paths = glob(f"{root}/*/{sample_type}/*/*.png")    
    all_image_lbls  = get_labels(all_image_paths)    

    tr_paths, test_paths, tr_lbls, test_lbls = train_test_split(all_image_paths, all_image_lbls, test_size=0.25, stratify=all_image_lbls, random_state=random_state)    
    vl_paths, ts_paths, vl_lbls, ts_lbls = train_test_split(test_paths, test_lbls, test_size=0.3, stratify=test_lbls, random_state=random_state)    

    return tr_paths, tr_lbls, vl_paths, vl_lbls, ts_paths, ts_lbls

def get_meta_data(root):

    # with open(f"{root}/position_ones_augmented.json", 'r', encoding='utf-8-sig') as f:
    with open(f"{root}/position.json", 'r', encoding='utf-8-sig') as f: bbox_data = json.load(f)

    bbox_dict = {}
    for item in bbox_data:
        fname = item['FileName']
        if "_normal" in fname:
            x, y, w, h = [item['x'], item['y'], item['w'], item['h']]                
        else:
            pos = item["Position"][0]                
            x, y, w, h = [pos['x'], pos['y'], pos['x'] + pos['w'], pos['y'] + pos['h']]           
        
        bbox_dict[fname] = [ x, y, w, h ]
    
    return bbox_dict

# def get_meta_data(root):

#     # with open(f"{root}/position_ones_augmented.json", 'r', encoding='utf-8-sig') as f:
#     with open(f"{root}/position_full.json", 'r', encoding='utf-8-sig') as f: bbox_data = json.load(f)
#     filtered_data = [d for d in bbox_data if "Position" in d]    

#     # Dynamically find maximum number of bboxes
#     max_abnormal = max(len(item["Position"]) for item in filtered_data)
#     max_normal = 1  # Normal images currently have 1 bbox each
#     MAX_BBOXES = max(max_abnormal, max_normal)
    
#     PAD_VALUE = [-1, -1, -1, -1]

#     def pad_bboxes(bboxes): return bboxes[:MAX_BBOXES] + [PAD_VALUE] * (MAX_BBOXES - len(bboxes))

#     bbox_dict = {}

#     # Process abnormal images
#     for item in bbox_data:

#         fname = item['FileName']
        
#         if "_normal" in fname: bbox_dict[fname] = item["bboxes"]
        
#         else:
#             bboxes = [
#                 [pos['x'], pos['y'], pos['x'] + pos['w'], pos['y'] + pos['h']]
#                 for pos in item["Position"]
#             ]
#             bbox_dict[fname] = pad_bboxes(bboxes)

#     return bbox_dict

# def get_meta_data(root):

#     # with open(f"{root}/position_ones_augmented.json", 'r', encoding='utf-8-sig') as f:
#     with open(f"{root}/abnormal/position.json", 'r', encoding='utf-8-sig') as f: abnormal_bbox_data = json.load(f)
#     with open(f"/vol0/nfs9/tileimage/new/25_05_30/position_ones.json", 'r', encoding='utf-8-sig') as f: normal_bbox_data = json.load(f)
    
#     abnormal_bbox_dict, normal_bbox_dict = {}, {}
#     for item in abnormal_bbox_data:
#         fname = item['FileName']                
#         pos = item["Position"][0]
#         x, y, w, h = [pos['x'], pos['y'], pos['x'] + pos['w'], pos['y'] + pos['h']]                        
#         abnormal_bbox_dict[fname] = [ x, y, w, h ]
    
#     for item in normal_bbox_data:        
#         fname = item['FileName']
#         if "_normal" in fname: 
#             x, y, w, h = [item['x'], item['y'], item['w'], item['h']]                    
#             normal_bbox_dict[fname] = [ x, y, w, h ]

#     return abnormal_bbox_dict, normal_bbox_dict

# def get_meta_data(root):
#     import json

#     with open(f"{root}/abnormal/position.json", 'r', encoding='utf-8-sig') as f:
#         abnormal_bbox_data = json.load(f)
#     with open(f"/vol0/nfs9/tileimage/new/25_05_30/position_ones.json", 'r', encoding='utf-8-sig') as f:
#         normal_bbox_data = json.load(f)

#     # Dynamically find maximum number of bboxes
#     max_abnormal = max(len(item["Position"]) for item in abnormal_bbox_data)
#     max_normal = 1  # Normal images currently have 1 bbox each
#     MAX_BBOXES = max(max_abnormal, max_normal)
    
#     PAD_VALUE = [-1, -1, -1, -1]

#     def pad_bboxes(bboxes):
#         return bboxes[:MAX_BBOXES] + [PAD_VALUE] * (MAX_BBOXES - len(bboxes))

#     abnormal_bbox_dict, normal_bbox_dict = {}, {}

#     # Process abnormal images
#     for item in abnormal_bbox_data:
#         fname = item['FileName']
#         bboxes = [
#             [pos['x'], pos['y'], pos['x'] + pos['w'], pos['y'] + pos['h']]
#             for pos in item["Position"]
#         ]
#         abnormal_bbox_dict[fname] = pad_bboxes(bboxes)

#     # Process normal images
#     for item in normal_bbox_data:
#         fname = item['FileName']
#         if "_normal" in fname:
#             x, y, w, h = item['x'], item['y'], item['w'], item['h']
#             normal_bbox_dict[fname] = pad_bboxes([[x, y, x + w, y + h]])

#     return abnormal_bbox_dict, normal_bbox_dict

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=None, gamma=2, reduction='mean'):
#         super().__init__()
#         self.alpha = alpha  # Your existing class weights
#         self.gamma = gamma
#         self.reduction = reduction
#         self.ce = nn.CrossEntropyLoss(weight=alpha, reduction='none')

#     def forward(self, inputs, targets):
#         ce_loss = self.ce(inputs, targets)
#         pt = torch.exp(-ce_loss)  # Probability of true class
#         focal_loss = (1 - pt)**self.gamma * ce_loss        
#         self.alpha = self.alpha.to(ce_loss.device)        
            
#         if self.alpha is not None:            
#             focal_loss *= self.alpha[targets]  # Apply class weights
            
#         return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.register_buffer('alpha', torch.tensor(alpha) if alpha is not None else None)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        logpt = F.log_softmax(inputs, dim=1)                    # (batch_size, n_classes)
        pt = logpt.exp()                                        # (batch_size, n_classes)
        logpt = logpt[range(inputs.shape[0]), targets]          # (batch_size,)
        pt = pt[range(inputs.shape[0]), targets]                # (batch_size,)
        focal_term = (1 - pt) ** self.gamma
        loss = -focal_term * logpt                              # (batch_size,)
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            loss = alpha[targets] * loss
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()