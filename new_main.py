import os
import torch
import pickle
import argparse
from glob import glob
import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from dataset import CellDataModule
from model import CellClassifier
from vis_original import DatasetVisualizer
from transformations import get_fts

class TileClassificationPipeline:
    def __init__(self, args):
        self.args = args
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.model_save_dir = os.path.join("checkpoints", self.args.data_name)
        self.save_data_dir = "saved_data"
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.save_data_dir, exist_ok=True)
        self.save_data_name = f"{self.args.run_name}_{self.args.data_name}"        
        self.dm = None
        self.model = None
        self.trainer = None
        self.callbacks = []

    def setup_datamodule(self):
        train_tfs, eval_tfs = get_fts(mean=self.mean, std=self.std)

        self.dm = CellDataModule(                        
            data_name=self.args.data_name,
            batch_size=self.args.batch_size,
            train_transform=train_tfs,
            eval_transform=eval_tfs,
            num_workers=4,
            persistent_workers=True,
        )
        self.dm.setup()
        
        self.class_counts = self.dm.train_dataset.class_counts
        self.class_names  = self.dm.train_dataset.class_names

        with open(f"{self.save_data_dir}/{self.save_data_name}_class_names.pkl", "wb") as f:
            pickle.dump(self.class_names, f)
        with open(f"{self.save_data_dir}/{self.save_data_name}_class_counts.pkl", "wb") as f:
            pickle.dump(self.class_counts, f)

    def setup_visualization(self):
        test_loader = self.dm.test_dataloader()
        torch.save(obj=test_loader, f=f"{self.save_data_dir}/{self.save_data_name}_test_dl.pth")

        DatasetVisualizer(
            train=self.dm.train_dataloader(),
            val=self.dm.val_dataloader(),
            test=test_loader,            
            data_name=self.args.data_name,
            run_name=self.args.run_name,
            save_dir="vis",
            num_samples=20
        ).run()

    def setup_model(self):
        self.model = CellClassifier(
            run_name=self.args.run_name,             
            class_counts=self.class_counts
        )

    def setup_callbacks(self):
        self.checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",            
            save_top_k=3,
            dirpath=self.model_save_dir,
            filename=f"{self.args.run_name}-best-{{epoch:02d}}-{{val_recall:.2f}}"
        )
        self.callbacks.append(self.checkpoint_callback)

        lr_monitor = LearningRateMonitor(logging_interval='epoch')        
        self.callbacks.append(lr_monitor)

        early_stop = EarlyStopping(
            monitor="val_loss",
            mode="min",
            min_delta=0.001,
            patience=self.args.patience,
            verbose=True
        )
        self.callbacks.append(early_stop)

    def setup_trainer(self):
        self.trainer = L.Trainer(
            accelerator="cuda",
            fast_dev_run=True,
            devices=2,
            strategy="ddp",
            max_epochs=self.args.max_epochs,
            precision="16-mixed",
            accumulate_grad_batches=self.args.accumulate_grad_batches,
            callbacks=self.callbacks
        )

    @staticmethod
    def get_best_ckpt(model_save_dir, metric, sample_type, run_name, data_name):
        highest_acc = 0
        ckpt_path = None
        pattern = f"{model_save_dir}/*{run_name}*.ckpt"        
        
        for file in glob(pattern):
            fname = os.path.splitext(os.path.basename(file))[0]
            acc_str = fname.split(f"{metric}=")[-1].split("-")[0]
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

        self.trainer.fit(self.model, self.dm)

        ckpt_path = self.get_best_ckpt(
            model_save_dir=self.model_save_dir,
            metric=self.args.metric,
            sample_type=self.args.sample_type,
            run_name=self.args.run_name,
            data_name=self.args.data_name
        )
        
        if ckpt_path:
            self.trainer.test(dataloaders=self.dm.test_dataloader(), ckpt_path=ckpt_path)
        else:
            print("No valid checkpoint found for testing!")

    @classmethod
    def parse_args(cls):
        import argparse
        parser = argparse.ArgumentParser(description="Cell Classification Training Pipeline")                
        parser.add_argument('--data_name', type=str, required=True, choices=['food', 'digit', 'art', 'kenya_food', 'retina', 'diabetic_retina'], help='Dataset name')        
        parser.add_argument('--run_name', type=str, required=True, help='Name for this training run')
        parser.add_argument('--metric', type=str, help='Metric to save best model')
        parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
        parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of training epochs')
        parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
        parser.add_argument('--accumulate_grad_batches', type=int, default=8, help='Number of batches for gradient accumulation')

        return parser.parse_args()


if __name__ == "__main__":    
    args = TileClassificationPipeline.parse_args()

    pipeline = TileClassificationPipeline(args)
    pipeline.run()
