from attention import HybridAttention
from utils import FocalLoss
import torch, timm
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners
import pytorch_lightning as L
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassSpecificity, MulticlassF1Score

class CellClassifier(L.LightningModule):
    def __init__(self, model_name, class_counts, num_classes=2, run_name=None):
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.run_name = run_name
        self.model = timm.create_model(model_name=self.model_name, pretrained=True, num_classes=self.num_classes)        

        # Validation metrics
        self.val_precision = MulticlassPrecision(num_classes=num_classes, average='macro')
        self.val_recall = MulticlassRecall(num_classes=num_classes, average='macro')
        self.val_specificity = MulticlassSpecificity(num_classes=num_classes, average='macro')
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')

        # Test metrics
        self.test_precision = MulticlassPrecision(num_classes=num_classes, average='macro')
        self.test_recall = MulticlassRecall(num_classes=num_classes, average='macro')
        self.test_specificity = MulticlassSpecificity(num_classes=num_classes, average='macro')
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')        

        # Loss and metric learning components
        self.miner = miners.MultiSimilarityMiner()
        self.contrastive_loss = losses.ContrastiveLoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()

        self.save_hyperparameters()

    def get_fms(self, fts): return fts[:, 0] if ("vit" in self.model_name or "eva" in self.model_name or "deit" in self.model_name) else torch.nn.functional.avg_pool2d(fts, kernel_size=(fts.shape[2], fts.shape[3])).squeeze(-1).squeeze(-1)

    def forward(self, x):
        
        features = self.model.forward_features(x)
        squeezed_features = self.get_fms(features)
        logits = self.model.forward_head(features)

        return logits, squeezed_features

    def training_step(self, batch, batch_idx):
        imgs, labels, _ = batch

        logits, features = self(imgs)

        # Use miner and contrastive loss on features
        pairs = self.miner(features, labels)
        loss_contrast = self.contrastive_loss(features, labels, pairs)

        # Cross-entropy classification loss
        loss_ce = self.ce_loss(logits, labels)

        total_loss = 0.6 * loss_contrast + 0.4 * loss_ce

        self.log_dict({
            "train_loss": total_loss,
            # "contrast_loss": loss_contrast,
            # "ce_loss": loss_ce
        }, prog_bar=True)

        return total_loss

    def on_train_epoch_end(self):
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        old_lr = getattr(self, 'last_lr', None)

        if old_lr is not None and current_lr != old_lr:
            print(f"\n🚀 Learning rate reduced from {old_lr:.2e} to {current_lr:.2e}")

        self.last_lr = current_lr

    def validation_step(self, batch, batch_idx):
        imgs, labels, _ = batch
        logits, _ = self(imgs)
        loss = self.ce_loss(logits, labels)
        acc = (logits.argmax(1) == labels).float().mean()
        preds = logits.argmax(dim=1)

        # Update metrics
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_specificity(preds, labels)
        self.val_f1(preds, labels)

        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True, sync_dist=True)

        return loss

    def on_validation_epoch_end(self):
        self.log_dict({
            "val_precision": self.val_precision.compute(),
            "val_recall": self.val_recall.compute(),
            "val_specificity": self.val_specificity.compute(),
            "val_f1": self.val_f1.compute()
        }, prog_bar=True, sync_dist=True)

        self.val_precision.reset()
        self.val_recall.reset()
        self.val_specificity.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        imgs, labels, _ = batch
        logits, _ = self(imgs)
        loss = self.ce_loss(logits, labels)
        acc = (logits.argmax(1) == labels).float().mean()
        preds = logits.argmax(1)

        self.test_precision(preds, labels)
        self.test_recall(preds, labels)
        self.test_specificity(preds, labels)
        self.test_f1(preds, labels)

        self.log_dict({
            "test_loss": loss,
            "test_acc": acc
        }, sync_dist=True)

        return loss

    def on_test_epoch_end(self):
        self.log_dict({
            "test_precision": self.test_precision.compute(),
            "test_recall": self.test_recall.compute(),
            "test_specificity": self.test_specificity.compute(),
            "test_f1": self.test_f1.compute()
        }, sync_dist=True)

        self.test_precision.reset()
        self.test_recall.reset()
        self.test_specificity.reset()
        self.test_f1.reset()

    def configure_optimizers(self):        
            
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.1,
                patience=3,
                threshold=0.01,
                threshold_mode='rel'
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1
        }
        return [optimizer], [lr_scheduler]    
