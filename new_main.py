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

        self.dm = None
        self.model = None
        self.trainer = None
        self.callbacks = []

    def setup_datamodule(self):
        train_tfs, eval_tfs = get_fts(mean=self.mean, std=self.std, im_size=args.im_size)

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
        self.class_names = self.dm.train_dataset.class_names

        ts_ds_im_paths = self.dm.test_dataset.im_paths
        torch.save(obj=ts_ds_im_paths, f=f"{self.save_data_dir}/{self.args.data_name}_ts_ds_im_paths.pt")
        
        with open(f"{self.save_data_dir}/{self.args.data_name}_class_names.pkl", "wb") as f:
            pickle.dump(self.class_names, f)
        with open(f"{self.save_data_dir}/{self.args.data_name}_class_counts.pkl", "wb") as f:
            pickle.dump(self.class_counts, f)


    def setup_visualization(self):        

        DatasetVisualizer(
            train=self.dm.train_dataloader(),
            val=self.dm.val_dataloader(),            
            data_name=self.args.data_name,
            run_name=self.args.run_name,
            save_dir="vis",
            num_samples=20
            ).run()

    def setup_callbacks(self, model_name):
        # Clear existing callbacks to avoid duplicates
        self.callbacks = []

        self.checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",            
            save_top_k=3,
            dirpath=os.path.join(self.model_save_dir, model_name),
            filename=f"{self.args.run_name}-best-{{epoch:02d}}-{{val_acc:.2f}}",
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
        fast_dev_run=False,
        devices=2,
        strategy="ddp",
        max_epochs=self.args.max_epochs,
        # precision="16-mixed",
        precision="32-true",        
        accumulate_grad_batches=self.args.accumulate_grad_batches,
        callbacks=self.callbacks
        )
    
    def run(self):
        self.setup_datamodule()
        self.setup_visualization()

        results = {}

        for model_name in self.args.model_names:
            print(f"\n\nStarting training with model: {model_name}\n\n")

            # Use model_name as run_name or part of run_name for logging and checkpoint names
            current_run_name = f"{self.args.run_name}_{model_name}"

            self.setup_model_with_name(model_name=model_name, run_name=current_run_name)
            self.setup_callbacks(model_name=model_name)
            self.setup_trainer()

            self.trainer.fit(self.model, self.dm)

            # Optionally test after training (uncomment if desired)
            # ckpt_path = self.get_best_ckpt(
            #     model_save_dir=self.model_save_dir,
            #     metric=self.args.metric,
            #     sample_type=None,  # pass if used in get_best_ckpt
            #     run_name=current_run_name,
            #     data_name=self.args.data_name
            # )
            # if ckpt_path:
            #     self.trainer.test(dataloaders=self.dm.test_dataloader(), ckpt_path=ckpt_path)

            results[model_name] = "Trained"

        # You can save or return results as needed

    def setup_model_with_name(self, model_name, run_name):
        self.model = CellClassifier(
            run_name=run_name,
            class_counts=self.class_counts,
            num_classes=len(self.class_names),
            # optionally pass model_name to your model if supported
        )

    # Modify setup_model to raise an error or delegate to the above, so you avoid redundancy
    def setup_model(self):
        raise NotImplementedError("Use setup_model_with_name() with a model name parameter.")

    # parse_args method add an argument for model_names list:
    @classmethod
    def parse_args(cls):
        import argparse
        parser = argparse.ArgumentParser(description="Cell Classification Training Pipeline")
        parser.add_argument('--data_name', type=str, required=True, choices=['food', 'digit', 'art', 'kenya_food', 'retina', 'diabetic_retina'], help='Dataset name')
        parser.add_argument('--run_name', type=str, required=True, help='Base run name for training')
        parser.add_argument('--model_names', nargs='+', default=["ecaresnet269d", "resnet152d", "resnext101_64x4d", "rexnet_300", "vit_base_patch16_224", "swinv2_cr_small_ns_224", "convnextv2_base", "mobilenetv3_large_100", "swin_base_patch4_window7_224", "convnext_large",  "efficientnetv2_rw_m", "deit_base_patch16_224"], help='List of model names')    
        parser.add_argument('--metric', type=str, help='Metric to save best model')
        parser.add_argument('--im_size', type=int, default=224, help='Image size')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
        parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of training epochs')
        parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
        parser.add_argument('--accumulate_grad_batches', type=int, default=8, help='Number of batches for gradient accumulation')
        return parser.parse_args()

if __name__ == "__main__":    
    args = TileClassificationPipeline.parse_args()

    pipeline = TileClassificationPipeline(args)
    pipeline.run()
