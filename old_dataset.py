import os, json, torch, random, numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L
from PIL import Image; from glob import glob
from torchvision import transforms
from sklearn.model_selection import train_test_split

def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CustomDataset(Dataset):
    def __init__(self, image_dir, sample_type, empty_bbox_type, transform=None, image_paths=None, labels=None, bbox_dict=None):
        
        self.image_dir = f"{image_dir}/{sample_type}"
        self.transform = transform
        self.class_names = {"normal": 0, "abnormal": 1}        
        json_file = f"{image_dir}/position_{empty_bbox_type}.json"
        print(f"Working with json file located in -> {json_file}")
        # Load json data
        with open(json_file, 'r', encoding='utf-8-sig') as f: bbox_data = json.load(f)  
        
        self.bbox_dict = {}
        for item in bbox_data:
            fname = item['FileName']
            if "_normal" in fname:
                x, y, w, h = [item['x'], item['y'], item['w'], item['h']]                
            else:
                pos = item["Position"][0]                
                x, y, w, h = [pos['x'], pos['y'], pos['x'] + pos['w'], pos['y'] + pos['h']]           
            
            self.bbox_dict[fname] = [ x, y, w, h ]        
        
        # self.image_paths = glob(f"{self.image_dir}/*/*.png")
        self.image_paths = glob(f"{image_dir}/*/{sample_type}/*/*.png")
        self.labels = [self.class_names[self.get_class_name(path)] for path in self.image_paths]       
        self.get_info()
    
    # def get_class_name(self, path): return os.path.dirname(path).split("/")[-1]
    def get_class_name(self, path): return os.path.dirname(path).split("/")[-3]    

    def get_info(self):
        
        self.class_counts = {}        
        for idx, im_path in enumerate(self.image_paths):
            # if idx == 2: break
            class_name = self.get_class_name(im_path)            
            if class_name not in self.class_counts: 
                self.class_counts[class_name] = 1
            else: self.class_counts[class_name] += 1                
        print(self.class_counts)

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]        
        label = self.labels[idx]        
        
        # base_name = os.path.splitext(os.path.basename(img_path))[0] if label == 1 else f"{os.path.splitext(os.path.basename(img_path))[0]}_normal"
        base_name = os.path.basename(img_path) if label == 1 else f"{os.path.splitext(os.path.basename(img_path))[0]}_normal"

        # Load image
        img = Image.open(img_path).convert('RGB')        

        # Get bboxes
        bboxes = self.bbox_dict.get(base_name, [])
        bboxes = torch.tensor(bboxes, dtype=torch.float32) # Shape: [N,4]        

        if self.transform: img = self.transform(img)

        return img, label, bboxes

class CellDataModule(L.LightningDataModule):
    def __init__(self, image_dir, sample_type, empty_bbox_type, 
                 batch_size=32, train_transform=None, eval_transform=None,
                 num_workers=4, persistent_workers=True):
        super().__init__()
        self.seed = 42
        self.image_dir = image_dir
        self.sample_type = sample_type
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.empty_bbox_type = empty_bbox_type

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

    def setup(self, stage=None, split=[0.65, 0.25, 0.1]):
        
        ds = CustomDataset(image_dir=self.image_dir, empty_bbox_type=self.empty_bbox_type, sample_type=self.sample_type, transform=self.eval_transform)        

        self.ds = ds        

        total_len = len(ds); tr_len = int(total_len * split[0]); vl_len = int(total_len * split[1]); ts_len = total_len - (tr_len + vl_len)
        from torch.utils.data import random_split
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset=ds, lengths=[tr_len, vl_len, ts_len])        
        
        print(f"There are {len(self.train_dataset)} number of images in train dataset") 
        print(f"There are {len(self.val_dataset)} number of images in validation dataset") 
        print(f"There are {len(self.test_dataset)} number of images in test dataset\n")      

    def train_dataloader(self): return self._create_dataloader(self.train_dataset)

    def val_dataloader(self): return self._create_dataloader(self.val_dataset)

    def test_dataloader(self): return self._create_dataloader(self.test_dataset)