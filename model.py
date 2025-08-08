from attention import HybridAttention
from utils import FocalLoss
import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners
import pytorch_lightning as L
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassSpecificity, MulticlassF1Score


class CustomEncoder(torch.nn.Module):
    def __init__(self, backbone="resnet50"):
        super().__init__()
        resnet = torch.hub.load('pytorch/vision', backbone, pretrained=True)
        self.feature_extractor = torch.nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        self.hybrid_att = HybridAttention(in_channels=2048)  # Attention after feature extraction

    def forward(self, x):
        features = self.feature_extractor(x)  # [B, 2048, H, W]
        att_features = self.hybrid_att(features)
        return att_features


class CellClassifier(L.LightningModule):
    def __init__(self, class_counts, num_classes=2, run_name=None, backbone="resnet50", feature_dim=2048):
        super().__init__()

        # Image encoder
        self.encoder = CustomEncoder(backbone=backbone)
        self.num_classes = num_classes
        self.run_name = run_name

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

        # Classification head (optional small MLP before classifier)
        self.classifier_head = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, num_classes)
        )

        # Loss and metric learning components
        self.miner = miners.MultiSimilarityMiner()
        self.contrastive_loss = losses.ContrastiveLoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()

        self.save_hyperparameters()

    def forward(self, x):
        features = self.encoder(x)  # [B, 2048, H, W]
        features = features.mean(dim=[2, 3])  # Global Average Pooling to [B, 2048]
        logits = self.classifier_head(features)
        return logits, features

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
        if self.run_name and "scheduler" in self.run_name:
            print("Using lr scheduler")
            optimizer = torch.optim.AdamW(self.parameters(), lr=3e-5)
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
        else:
            print("Not using lr scheduler")
            return torch.optim.AdamW(self.parameters(), lr=3e-5)
