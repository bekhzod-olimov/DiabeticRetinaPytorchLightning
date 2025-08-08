import os, torch, pickle, argparse
from glob import glob
import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
# from new_dataset import CellDataModule
from old_dataset import CellDataModule
from model import CellClassifier
# from vis_original import DatasetVisualizer
from vis import DatasetVisualizer
from transformations import get_fts
from utils import get_splits, get_meta_data

class TileClassificationPipeline:
    def __init__(self, args):
        # Initialize parameters from args
        self.args = args
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.model_save_dir = "checkpoints/"        
        self.save_data_dir  = "saved_data"
        os.makedirs(self.model_save_dir, exist_ok=True); os.makedirs(self.save_data_dir, exist_ok=True)
        self.save_data_name = f"{self.args.run_name}_{self.args.sample_type}_{self.args.data_name}"        
        
        # Initialize components
        self.dm = None
        self.model = None
        self.trainer = None
        self.callbacks = []

    def setup_datamodule(self):
        train_tfs, eval_tfs = get_fts(mean=self.mean, std=self.std)  

        tr_paths, tr_lbls, vl_paths, vl_lbls, ts_paths, ts_lbls = get_splits(root = self.args.root_dir, sample_type = self.args.sample_type, random_state=42)
        bbox_dict = get_meta_data(root = self.args.root_dir)
                
        self.dm = CellDataModule(
            tr_paths, tr_lbls, 
            vl_paths, vl_lbls, 
            ts_paths, ts_lbls,
            bbox_dict = bbox_dict,
            sample_type=self.args.sample_type,
            batch_size=self.args.batch_size,
            train_transform=train_tfs,
            eval_transform=eval_tfs
        )
        self.dm.setup()
        print(f"self.class_names -> {self.dm.class_names}")
        with open(f"{self.save_data_dir}/{self.save_data_name}_class_names.pkl", "wb") as f: pickle.dump(self.dm.class_names, f)        

    def setup_visualization(self):        
        test_loader = self.dm.test_dataloader()
        torch.save(obj=test_loader, f=f"{self.save_data_dir}/{self.save_data_name}_test_dl.pth")

        DatasetVisualizer(
            data=self.dm.ds,      
            run_name=self.args.run_name,
            sample_type=self.args.sample_type,
            data_name=self.args.data_name,
            save_dir="vis",
            num_samples=20
        ).run()

    def setup_model(self): self.model = CellClassifier(run_name=self.args.run_name)

    def setup_callbacks(self):
        # Model checkpoint callback
        # self.checkpoint_callback = ModelCheckpoint(
        #     monitor="val_loss",
        #     mode="min",
        #     dirpath=self.model_save_dir,
        #     filename=f"{self.args.run_name}_{self.args.sample_type}_{self.args.data_name}-best-{{epoch:02d}}-{{val_acc:.2f}}"
        # )

        self.checkpoint_callback = ModelCheckpoint(
            monitor="val_recall",
            mode="max",
            dirpath=self.model_save_dir,
            filename=f"{self.args.run_name}_{self.args.sample_type}_{self.args.data_name}-best-{{epoch:02d}}-{{val_recall:.2f}}"
        )
        self.callbacks.append(self.checkpoint_callback)

        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval='epoch' if self.args.run_name == "scheduler" else 'step')
        self.callbacks.append(lr_monitor)

        # Early stopping
        early_stop = EarlyStopping(
            monitor="val_recall",
            mode="max",
            min_delta=0.001,
            patience=self.args.patience,
            verbose=True
        )
        self.callbacks.append(early_stop)

    def setup_trainer(self):
        self.trainer = L.Trainer(
            accelerator="gpu",
            fast_dev_run=True,
            devices=1,
            strategy="ddp",
            max_epochs=self.args.max_epochs,
            precision="16-mixed",
            accumulate_grad_batches=self.args.accumulate_grad_batches,
            callbacks=self.callbacks
        )

    @staticmethod
    def get_best_ckpt(model_save_dir, sample_type, run_name, data_name):
        highest_acc = 0
        ckpt_path = None
        pattern = f"{model_save_dir}/*{run_name}*{sample_type}*{data_name}*.ckpt"
        
        for file in glob(pattern):
            fname = os.path.splitext(os.path.basename(file))[0]
            acc_str = fname.split("val_acc=")[-1].split("-")[0]
            try:
                acc = float(acc_str)
                if acc > highest_acc:
                    highest_acc = acc
                    ckpt_path = file
            except ValueError:
                continue
                
        return ckpt_path

    def run(self):
        self.setup_datamodule()
        self.setup_visualization()
        self.setup_model()
        self.setup_callbacks()
        self.setup_trainer()

        # Training
        self.trainer.fit(self.model, self.dm)

        # Testing with best checkpoint
        ckpt_path = self.get_best_ckpt(
            self.model_save_dir,
            self.args.sample_type,
            self.args.run_name,
            self.args.data_name
        )
        
        if ckpt_path:
            self.trainer.test(dataloaders=self.dm.test_dataloader(), ckpt_path=ckpt_path)
        else:
            print("No valid checkpoint found for testing!")

    @classmethod
    def parse_args(cls):
        parser = argparse.ArgumentParser(description="Cell Classification Training Pipeline")
        parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing image data')
        parser.add_argument('--data_name', type=str, required=True, help='Dataset version/identifier')
        parser.add_argument('--sample_type', type=str, required=True, choices=['LBC', 'PAP'], help='Sample type for processing')
        parser.add_argument('--run_name', type=str, required=True, help='Name for this training run')
        parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
        parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of training epochs')
        parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
        parser.add_argument('--accumulate_grad_batches', type=int, default=8, help='Number of batches for gradient accumulation')

        return parser.parse_args()

if __name__ == "__main__":    
    
    args = TileClassificationPipeline.parse_args()
    pipeline = TileClassificationPipeline(args).run() 
