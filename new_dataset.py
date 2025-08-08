import os, torch, random, numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L
from PIL import Image
from utils import get_class_name

class CustomDataset(Dataset):
    # def __init__(self, transform=None, image_paths=None, labels=None, abnormal_bbox_dict=None, normal_bbox_dict=None, class_names = {"normal": 0, "abnormal": 1}):        
    def __init__(self, transform=None, image_paths=None, labels=None, bbox_dict=None, class_names = {"normal": 0, "abnormal": 1}):        
        
        self.image_paths = image_paths
        self.labels = labels
        # self.abnormal_bbox_dict = abnormal_bbox_dict;  self.normal_bbox_dict = normal_bbox_dict
        self.bbox_dict = bbox_dict
        self.transform = transform
        self.class_names = class_names        
        self.get_info()  

    def get_info(self):
        
        self.class_counts = {}        
        for idx, im_path in enumerate(self.image_paths):
            # if idx == 2: break            
            class_name = get_class_name(im_path)            
            if class_name not in self.class_counts: 
                self.class_counts[class_name] = 1
            else: self.class_counts[class_name] += 1                
        print(self.class_counts)

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]        
        label = self.class_names[self.labels[idx]]        
        
        # if "_180" in img_path:
        #     if label == 1:
        #         base_name = os.path.splitext(os.path.basename(img_path))[0]
        #     else:
        #         base_name = os.path.splitext(os.path.basename(img_path))[0]
        #         fname, count, _ = base_name.split("_")
        #         base_name = f"{fname}_{count}_normal_180"
        # else: 
        #     base_name = os.path.splitext(os.path.basename(img_path))[0] if label == 1 else f"{os.path.splitext(os.path.basename(img_path))[0]}_normal"               
        
        
        # if label == 1:     
        #     base_name = os.path.basename(img_path)
        #     self.bbox_dict = self.abnormal_bbox_dict                        
        #     # if base_name not in self.bbox_dict: print(f"normal img_path {base_name}")
        # else:
        #     base_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_normal"
        #     self.bbox_dict = self.normal_bbox_dict                       
        #     # print(f"{list(self.normal_bbox_dict.keys())[:10]}")

        # Load image
        img = Image.open(img_path).convert('RGB')
        base_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_normal" if label == 0 else os.path.basename(img_path)

        # Get bboxes        
        bboxes = self.bbox_dict.get(base_name, [])
        bboxes = torch.tensor(bboxes, dtype=torch.float32) # Shape: [N,4]                

        if self.transform: img = self.transform(img)

        return img, label, bboxes

# class CellDataModule(L.LightningDataModule):
#     def __init__(self, tr_paths, tr_lbls, vl_paths, vl_lbls, 
#                  ts_paths, ts_lbls, abnormal_bbox_dict, normal_bbox_dict, sample_type,
#                  batch_size=32, train_transform=None, eval_transform=None,A
#                  num_workers=4, persistent_workers=True):
#         super().__init__()
#         self.seed = 42        
#         self.sample_type = sample_type
#         self.batch_size = batch_size
#         self.train_transform = train_transform
#         self.eval_transform = eval_transform
#         self.num_workers = num_workers
#         self.persistent_workers = persistent_workers
#         self.tr_paths, self.tr_lbls = tr_paths, tr_lbls
#         self.vl_paths, self.vl_lbls = vl_paths, vl_lbls 
#         self.ts_paths, self.ts_lbls = ts_paths, ts_lbls
#         self.abnormal_bbox_dict = abnormal_bbox_dict
#         self.normal_bbox_dict = normal_bbox_dict

#     def _seed_worker(self, worker_id):
#         worker_seed = torch.initial_seed() % 2**32
#         np.random.seed(worker_seed)
#         random.seed(worker_seed)
    
#     def _create_dataloader(self, dataset):
#         generator = torch.Generator()
#         generator.manual_seed(self.seed)
        
#         return DataLoader(
#             dataset,
#             batch_size=self.batch_size,
#             shuffle=(dataset == self.train_dataset),
#             num_workers=self.num_workers,
#             worker_init_fn=self._seed_worker,
#             generator=generator,
#             persistent_workers=self.persistent_workers,
#             pin_memory=True
#         )

#     def setup(self, stage=None):        
        
#         self.train_dataset = CustomDataset(image_paths=self.tr_paths, labels=self.tr_lbls, abnormal_bbox_dict=self.abnormal_bbox_dict, normal_bbox_dict = self.normal_bbox_dict, transform=self.train_transform)
#         self.val_dataset = CustomDataset(image_paths=self.vl_paths, labels=self.vl_lbls, abnormal_bbox_dict=self.abnormal_bbox_dict, normal_bbox_dict = self.normal_bbox_dict,  transform=self.eval_transform)
#         self.test_dataset = CustomDataset(image_paths=self.ts_paths, labels=self.ts_lbls, abnormal_bbox_dict=self.abnormal_bbox_dict, normal_bbox_dict = self.normal_bbox_dict, transform=self.eval_transform)
#         self.class_names = self.train_dataset.class_names
               
#         print(f"There are {len(self.train_dataset)} number of images in train dataset") 
#         print(f"There are {len(self.val_dataset)} number of images in validation dataset") 
#         print(f"There are {len(self.test_dataset)} number of images in test dataset\n")      

#     def train_dataloader(self): return self._create_dataloader(self.train_dataset)

#     def val_dataloader(self): return self._create_dataloader(self.val_dataset)

#     def test_dataloader(self): return self._create_dataloader(self.test_dataset) 

class CellDataModule(L.LightningDataModule):
    def __init__(self, tr_paths, tr_lbls, vl_paths, vl_lbls, 
                 ts_paths, ts_lbls, bbox_dict, sample_type,
                 batch_size=32, train_transform=None, eval_transform=None,
                 num_workers=4, persistent_workers=True):
        super().__init__()
        self.seed = 42        
        self.sample_type = sample_type
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.tr_paths, self.tr_lbls = tr_paths, tr_lbls
        self.vl_paths, self.vl_lbls = vl_paths, vl_lbls 
        self.ts_paths, self.ts_lbls = ts_paths, ts_lbls
        self.bbox_dict = bbox_dict        

    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    def _create_dataloader(self, dataset):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(dataset == self.train_dataset),
            num_workers=self.num_workers,
            worker_init_fn=self._seed_worker,
            generator=generator,
            persistent_workers=self.persistent_workers,
            pin_memory=True
        )

    def setup(self, stage=None):        
        
        self.train_dataset = CustomDataset(image_paths=self.tr_paths, labels=self.tr_lbls, bbox_dict=self.bbox_dict, transform=self.train_transform)
        self.val_dataset = CustomDataset(image_paths=self.vl_paths, labels=self.vl_lbls, bbox_dict=self.bbox_dict,  transform=self.eval_transform)
        self.test_dataset = CustomDataset(image_paths=self.ts_paths, labels=self.ts_lbls, bbox_dict=self.bbox_dict, transform=self.eval_transform)
        self.class_names = self.train_dataset.class_names
               
        print(f"There are {len(self.train_dataset)} number of images in train dataset") 
        print(f"There are {len(self.val_dataset)} number of images in validation dataset") 
        print(f"There are {len(self.test_dataset)} number of images in test dataset\n")      

    def train_dataloader(self): return self._create_dataloader(self.train_dataset)

    def val_dataloader(self): return self._create_dataloader(self.val_dataset)

    def test_dataloader(self): return self._create_dataloader(self.test_dataset)