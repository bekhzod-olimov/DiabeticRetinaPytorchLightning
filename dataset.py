import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split
import random
import numpy as np


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CustomDataset(Dataset):
    def __init__(
        self,        
        data_name,
        transform=None,
        class_names=None,
        image_paths=None,
        labels=None,
    ):
        
        self.data_name = data_name
        self.transform = transform

        # Set class_names dynamically or from provided        
        if class_names is not None:
            self.class_names = class_names
        else:
            self.class_names = {}
        
        # Initialize dataset structure based on data_name
        self._set_dataset_structure()
        

        if image_paths is not None and labels is not None:
            # Use provided split data
            self.image_paths = image_paths
            self.labels = labels
            self.get_info()
            print(f"self.class_counts -> {self.class_counts}")
        else:
            self.get_im_paths()
            self.get_im_labels()
            self.get_info()

    def _set_dataset_structure(self):
        data_root_dir = "/home/bekhzod/Desktop/backup/image_classification_project_datasets"

        if self.data_name == "food":
            self.root = os.path.join(data_root_dir, "street-food-image-classification")
            self.image_dir = f"{self.root}/train_images"
            self.image_paths = glob(f"{self.image_dir}/*/*.jpg")
            self._build_cls_names_from_dirs()
            self.labels = []
        elif self.data_name == "digit":
            self.root = os.path.join(data_root_dir, "xpc-team-digit-recognizer")
            meta_data = pd.read_csv(f"{self.root}/train.csv")
            self.labels = list(meta_data["label"])
            self.images = meta_data.drop(columns=["label"]).values.astype("uint8")
            self.image_paths = list(range(len(self.labels)))  # indices as "paths"
            if not self.class_names:
                self.class_names = {str(i): i for i in set(self.labels)}
        elif self.data_name in ["art", "diabetic_retina"]:
            self.root = (
                os.path.join(data_root_dir, "boolart-image-classification")
                if self.data_name == "art"
                # else "/home/bekhzod/Desktop/backup/image_classification_project_datasets/diabetic_retina_new_no_tfs"
                else "/home/super/Desktop/bekhzod/kaggle/diabetic_retina_new_no_tfs"
            )
            target_name = "target" if self.data_name == "art" else "diagnosis"
            id_name = "id" if self.data_name == "art" else "id_code"
            im_dir_name = "train_image" if self.data_name == "art" else "train"
            ext_name = "jpg" if self.data_name == "art" else "png"

            meta_data = pd.read_csv(f"{self.root}/train.csv")
            self.labels = list(meta_data[target_name])
            self.path_with_labels = {id: int(label) for id, label in zip(list(meta_data[id_name]), list(meta_data[target_name]))}
            ids = list(meta_data[id_name])
            self.image_paths = [os.path.join(self.root, im_dir_name, f"{id}.{ext_name}") for id in ids]            
            if not self.class_names:
                self.class_names = {i: i for i in set(self.labels)}
        elif self.data_name == "kenya_food":
            self.root = "food"
            meta_data = pd.read_csv(f"{self.root}/train.csv")
            test_data_ids = pd.read_csv(f"{self.root}/test.csv")["id"]
            self.labels = list(meta_data["class"])
            ids = list(meta_data["id"])
            self.path_with_labels = {id: label for id, label in zip(ids, self.labels)}
            dir_name = "images"
            self.image_paths = [os.path.join(self.root, dir_name, dir_name, f"{id}.jpg") for id in ids]
            self.test_im_paths = [os.path.join(self.root, dir_name, dir_name, f"{id}.jpg") for id in test_data_ids]
            if not self.class_names:
                self.class_names = {i: idx for idx, i in enumerate(set(self.labels))}
        elif self.data_name == "retina":
            self.root = os.path.join(data_root_dir, "retina")
            self.image_dir = os.path.join(self.root, "retina-train")
            self.image_paths = glob(f"{self.image_dir}/*.jpeg")
            if not self.class_names:
                self.class_names = {}
        else:
            raise ValueError(f"Unsupported data_name: {self.data_name}")

    def _build_cls_names_from_dirs(self):
        dirs = []
        if os.path.isdir(self.image_dir):
            dirs = [d for d in os.listdir(self.image_dir) if os.path.isdir(os.path.join(self.image_dir, d))]
        dirs = sorted(dirs)
        self.class_names = {cls_name: idx for idx, cls_name in enumerate(dirs)}
        if not self.class_names:
            raise ValueError(f"No class directories found in {self.image_dir}.")
        print(f"Detected classes: {self.class_names}")

    def get_class_name(self, path):
        if self.data_name == "digit":
            return str(self.labels[path])
        elif self.data_name == "food":
            return os.path.basename(os.path.dirname(path))
        elif self.data_name == "retina":
            return os.path.splitext(os.path.basename(path))[0].split("_")[-1]
        elif self.data_name in ["art", "kenya_food"]:
            key = self.get_id(path) if self.data_name == "art" else os.path.splitext(os.path.basename(path))[0]
            return self.path_with_labels[key]
        elif self.data_name == "diabetic_retina":
            key = os.path.splitext(os.path.basename(path))[0]
            return self.path_with_labels[key]
        else:
            return os.path.dirname(path).split(os.sep)[-2]

    def get_id(self, path):
        return int(os.path.splitext(os.path.basename(path))[0])

    def get_im_paths(self):
        if hasattr(self, "image_paths") and self.image_paths:
            return
        self.image_paths = [path for path in glob(f"{self.root}/*/*/*/*.png") if self.sample_type in path]

    def get_im_labels(self):
        if hasattr(self, "labels") and self.labels:
            return
        self.labels = []
        for p in self.image_paths:
            cls_name = self.get_class_name(p)
            if isinstance(cls_name, int):
                label_idx = cls_name
            else:
                label_idx = self.class_names.get(cls_name)
                if label_idx is None:
                    label_idx = len(self.class_names)
                    self.class_names[cls_name] = label_idx
            self.labels.append(label_idx)

    def get_info(self):
        self.class_counts = {}
        for idx, im_path in enumerate(self.image_paths):
            class_name = self.get_class_name(im_path)
            if class_name not in self.class_counts:
                self.class_counts[class_name] = 1
            else:
                self.class_counts[class_name] += 1
        print(f"Class counts: {self.class_counts}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label, 0

    @classmethod
    def stratified_split(cls, data_name, train_transform=None, eval_transform = None, train_ratio=0.8, val_ratio=0.2, random_state=42):
        
        full_dataset = cls(data_name=data_name, transform=eval_transform)

        train_paths, val_paths, train_labels, val_labels = train_test_split(
            full_dataset.image_paths,
            full_dataset.labels,
            stratify=full_dataset.labels,
            test_size=1 - train_ratio,
            random_state=random_state,
        )        

        train_dataset = cls(            
            data_name=data_name,
            transform=train_transform,
            image_paths=train_paths,
            labels=train_labels,
            class_names=full_dataset.class_names,
        )

        val_dataset = cls(
                        data_name=data_name,
            transform=eval_transform,
            image_paths=val_paths,
            labels=val_labels,
            class_names=full_dataset.class_names,
        )

        if data_name == "food": test_im_paths = glob(f"{train_dataset.root}/test_images/*.jpg")    
        elif data_name == "art": test_im_paths = glob(f"{train_dataset.root}/test_image/*.jpg")
        elif data_name == "diabetic_retina": print(train_dataset.root); test_im_paths = glob(f"{train_dataset.root}/test/*.png")
        elif data_name == "retina": test_im_paths = glob(f"{train_dataset.root}/retina-test/*.jpeg")
        elif data_name == "digit":
            meta_data = pd.read_csv(f"{train_dataset.root}/test.csv")
            test_im_paths = meta_data.values.astype("uint8")            
        elif data_name == "kenya_food":
            test_im_paths = train_dataset.test_im_paths
        
        class CustomTestDataset(Dataset):

            def __init__(self, im_paths, transformations):
                self.im_paths = im_paths
                self.transformations = transformations
            
            def __len__(self): return len(self.im_paths)
            
            def __getitem__(self, idx):
                im_path = self.im_paths[idx]
                if data_name in ["food", "art", "diabetic_retina", "kenya_food", "retina"]:
                    im = Image.open(im_path).convert("RGB")
                elif data_name == "digit":
                    im = Image.fromarray(self.im_paths[idx].reshape(28, 28), mode="L").convert("RGB")
                if self.transformations:
                    im = self.transformations(im)
                return {"im_path": im_path, "im": im}
            
        test_dataset = CustomTestDataset(test_im_paths, eval_transform)

        return train_dataset, val_dataset, test_dataset

class CellDataModule(L.LightningDataModule):
    def __init__(
        self,        
        data_name,
        batch_size=32,
        train_transform=None,
        eval_transform=None,
        num_workers=4,
        persistent_workers=True,
    ):
        super().__init__()
        self.seed = 42
        
        self.data_name = data_name
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

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
            pin_memory=True,
        )

    def setup(self, stage=None):
        self.train_dataset, self.val_dataset, self.test_dataset = CustomDataset.stratified_split(            
            data_name=self.data_name,
            train_transform=self.train_transform,
            eval_transform=self.eval_transform
        )
        self.class_names = self.train_dataset.class_names
        print(f"There are {len(self.train_dataset)} images in train dataset")
        print(f"There are {len(self.val_dataset)} images in validation dataset")
        print(f"There are {len(self.test_dataset)} images in test dataset\n")

        if self.eval_transform:
            self.val_dataset.transform = self.eval_transform            

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset)   
    
    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset)   
