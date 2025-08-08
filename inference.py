import os
import argparse
import torch
import pickle
import random
import numpy as np
import seaborn as sns
from time import time
from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from model import CellClassifier
from main import TileClassificationPipeline
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinarySpecificity

class Denormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

class ModelInferenceVisualizer:
    def __init__(self, model, device, results_dir, save_name, sample_type, run_name, save_wrong, 
                 data_name, class_names=None, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        
        self.accuracy = BinaryAccuracy().to(device)
        self.precision = BinaryPrecision().to(device)
        self.recall = BinaryRecall().to(device)
        self.f1 = BinaryF1Score().to(device)
        self.specificity = BinarySpecificity().to(device)
        self.denormalize = Denormalize(mean, std)
        self.model = model
        self.device = device        
        self.results_dir = results_dir
        self.class_names = class_names        
        self.save_name = save_name
        self.sample_type = sample_type
        self.run_name = run_name
        self.model.eval()
        os.makedirs(self.results_dir, exist_ok=True)
        self.save_wrong = save_wrong        
        if self.save_wrong:            
            self.wrong_preds_dir = os.path.join("wrong_preds", data_name)
            self.wrong_count  = 0

    def tensor_to_image(self, tensor):
        tensor = self.denormalize(tensor)
        tensor = tensor.permute(1, 2, 0)
        return (tensor.cpu().numpy() * 255).astype(np.uint8)

    def plot_value_array(self, logits, gt, class_names):
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1)
        
        plt.grid(visible=True)
        plt.xticks(range(len(class_names)), class_names, rotation='vertical')
        plt.yticks(np.arange(0.0, 1.1, 0.1))
        bars = plt.bar(range(len(class_names)), [p.item() for p in probs[0]], color="#777777")
        plt.ylim([0, 1])
        if pred_class.item() == gt:
            bars[pred_class].set_color('green')
        else:
            bars[pred_class].set_color('red')

    def generate_cam_visualization(self, image_tensor):
        cam = GradCAMPlusPlus(model=self.model, target_layers=[self.model.encoder.feature_extractor[-1][-1].conv3], use_cuda=self.device == "cuda")
        grayscale_cam = cam(input_tensor=image_tensor.unsqueeze(0))[0, :]
        return grayscale_cam

    def to_device(self, batch): return batch[0].to(self.device), batch[1].to(self.device)

    def save_wrong_prediction(self, ims, logits, preds, lbls): 
        
        os.makedirs(self.wrong_preds_dir, exist_ok=True)        
        current_spt_dir = os.path.join(self.wrong_preds_dir, self.sample_type)
        os.makedirs(current_spt_dir, exist_ok=True)
        current_run_dir = os.path.join(current_spt_dir, self.run_name)        
        os.makedirs(current_run_dir, exist_ok=True)        
        os.makedirs(os.path.join(current_run_dir, "normal"), exist_ok=True)
        os.makedirs(os.path.join(current_run_dir, "abnormal"), exist_ok=True)

        for im, pred, logit, lbl in zip(ims, preds, logits, lbls):
            if pred != lbl:
                class_name = self.class_names[lbl.item()]
                self.wrong_count += 1                
                probs = torch.nn.functional.softmax(logit, dim = 0)                
                res   = probs[pred.item()]                
                plt.imshow(self.tensor_to_image(im.squeeze()))
                plt.axis("off")
                plt.title(f"GT: {class_name} | PR: {self.class_names[pred.item()]} | {(res * 100):.2f}%")
                plt.savefig(os.path.join(current_run_dir, class_name, f"wrong_{self.wrong_count}.png"))

    def visualize(self, num_images, rows):
        count = 1
        plt.figure(figsize=(30, 15))
        indices = [random.randint(0, len(self.images)-1) for _ in range(num_images)]
        print("Visualizing probability comparison and GradCAM plots...")
        for idx, index in enumerate(indices):
            im = self.tensor_to_image(self.images[index].squeeze())
            pred_idx = self.preds[index]
            gt_idx = self.lbls[index]

            plt.subplot(rows, 2*num_images//rows, count)
            count += 1
            plt.imshow(im, cmap="gray")
            plt.axis("off")

            grayscale_cam = self.generate_cam_visualization(self.images[index])
            visualization = show_cam_on_image(im/255, grayscale_cam, image_weight=0.5, use_rgb=True)
            plt.imshow(visualization, alpha=0.9, cmap='jet')
            plt.axis("off")

            logits = self.logitss[index]
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            plt.subplot(rows, 2*num_images//rows, count)
            count += 1
            self.plot_value_array(logits=logits, gt=gt_idx, class_names=self.class_names)

            if self.class_names:
                gt_name = self.class_names[gt_idx]
                pred_name = self.class_names[pred_idx]
                color = "green" if gt_name == pred_name else "red"
                plt.title(f"GT: {gt_name} | PRED: {pred_name}", color=color)
            plt.savefig(f"{self.results_dir}/{self.save_name}_results.png")

        print("Visualizing confusion matrix...")
        plt.figure(figsize=(20, 10))
        cm = confusion_matrix([all_lbl.cpu() for all_lbl in self.all_lbls], [all_pred.cpu() for all_pred in self.all_preds])
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")        
        plt.savefig(f"{self.results_dir}/{self.save_name}_confusion_matrix.png")

    def infer_and_visualize(self, test_dl, num_images, rows): 
        infer_start = time(); self.infer(test_dl=test_dl); print(f"Inference is completed in {(time() - infer_start):.3f} secs.")
        vis_start   = time(); self.visualize(num_images, rows); print(f"Visualization is completed in {(time() - vis_start):.3f} secs.")
        print(f"Inference and visualization processes are completed in {(time() - infer_start):.3f} secs.")
    
    def infer(self, test_dl):
        
        # Reset metrics
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.specificity.reset()

        self.preds, self.images, self.lbls, self.logitss, self.all_preds, self.all_lbls = [], [], [], [], [], []
        accuracy, total = 0, 0

        with torch.no_grad():
            for idx, batch in tqdm(enumerate(test_dl), desc="Inference"):
                # if idx == 1: break
                im, gt = self.to_device(batch)
                with torch.amp.autocast("cuda"): logits = self.model(im)
                pred_class = torch.argmax(logits, dim=1)

                # Update metrics on GPU
                self.accuracy(pred_class, gt)
                self.precision(pred_class, gt)
                self.recall(pred_class, gt)
                self.f1(pred_class, gt)
                self.specificity(pred_class, gt)

                if self.save_wrong: self.save_wrong_prediction(ims=im, preds=pred_class, logits=logits, lbls=gt)
                total += im.shape[0]                
                accuracy += (pred_class == gt).sum().item()
                self.images.append(im[0].cpu())
                self.logitss.append(logits[0])
                self.preds.append(pred_class[0].item())
                self.lbls.append(gt[0].item())
                self.all_preds.extend(pred_class)
                self.all_lbls.extend(gt)

        print(f"\nF1 score          of the model on the test data -> {(self.f1.compute().item()):.3f}")
        print(f"Recall score      of the model on the test data -> {(self.recall.compute().item()):.3f}")
        print(f"Precision score   of the model on the test data -> {(self.precision.compute().item()):.3f}")        
        print(f"Specificity score of the model on the test data -> {(self.specificity.compute().item()):.3f}")
        print(f"Accuracy score    of the model on the test data -> {(self.accuracy.compute().item()):.3f}")
        print(f"Accuracy score    of the model on the test data -> {(accuracy / total):.3f}\n")

def parse_args():
    parser = argparse.ArgumentParser(description='Model Inference Configuration')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference')
    parser.add_argument('--results_dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--run_name', type=str, default='scheduler', help='Name of the training run')
    parser.add_argument('--empty_bbox_type', type=str, required=True, choices=['zeros', 'ones'], help='Empty bbox type')
    parser.add_argument('--saved_data_dir', type=str, default='saved_data', help='Directory for saved data')
    parser.add_argument('--metric', type=str, help='Metric to save best model')
    parser.add_argument('--ckpt_path', type=str, help='Checkpoint path')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Checkpoint directory')    
    parser.add_argument('--data_name', type=str, default='25_05_09_1024', help='Dataset identifier')
    parser.add_argument('--sample_type', type=str, default='LBC', help='Sample type specification')
    parser.add_argument('--save_wrong', action='store_true', help='Whether or not to save wrong predictions')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    save_name = f"{args.run_name}_{args.sample_type}_{args.data_name}"
    
    ts_dl_path = f"{args.saved_data_dir}/{save_name}_test_dl.pth"    
    # ts_dl_path = "saved_data/scheduler_bbox_ones_model_ones_minimize_loss_new_method_PAP_25_05_15_test_dl.pth"
    print(f"ts_dl_path -> {ts_dl_path}")    
    ts_dl = torch.load(ts_dl_path, weights_only=False)
    ckpt_path = args.ckpt_path
    # ckpt_path = TileClassificationPipeline.get_best_ckpt(
    #     model_save_dir=args.save_dir,
    #     metric=args.metric,
    #     sample_type=args.sample_type,
    #     run_name=args.run_name,
    #     data_name=args.data_name
    # )
    print(f"ckpt_path -> {ckpt_path}")    
    
    # with open(f"{args.saved_data_dir}/{save_name}_class_names.pkl", "rb") as fp: classes = pickle.load(fp)
    # with open(f"{args.saved_data_dir}/{save_name}_class_counts.pkl", "rb") as fp: class_counts = pickle.load(fp)
    class_counts = {'normal': 95394, 'abnormal': 25095}
    classes = {'normal': 0, 'abnormal': 1}
    
    model = CellClassifier(run_name=args.run_name, empty_bbox_type=args.empty_bbox_type, class_counts=class_counts).to(args.device)
    print("Before:", model.classifier.weight[0][:5])    
    model.load_state_dict(torch.load(ckpt_path, weights_only=False)["state_dict"], strict=False)    
    print("After:", model.classifier.weight[0][:5])    
    
    inference_visualizer = ModelInferenceVisualizer(
        model=model,
        device=args.device,
        save_wrong=args.save_wrong,
        sample_type=args.sample_type,
        run_name=args.run_name,
        data_name=args.data_name,
        save_name=save_name,
        results_dir=args.results_dir,
        class_names=list(classes.keys())
    )
    
    inference_visualizer.infer_and_visualize(ts_dl, num_images=20, rows=4)
