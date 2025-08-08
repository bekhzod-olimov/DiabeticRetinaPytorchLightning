import os, torch, random, numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms as T

class DatasetVisualizer:
    def __init__(self, train, val, run_name, data_name, save_dir, splits=['train', 'val'], num_samples=20):        
        
        self.train, self.val = train, val
        self.save_dir = save_dir
        self.run_name, self.data_name = run_name, data_name 
        self.splits = splits
        self.num_samples = num_samples
        os.makedirs(self.save_dir, exist_ok = True)        
        self.datasets = {
                self.splits[0]: self.train.dataset,
                self.splits[1]: self.val.dataset,                
            }
        self.colors = ["darkorange", "seagreen", "salmon"]

    def denormalize(self, tensor):
        """Reverse normalization for visualization"""
        tensor = tensor.clone().cpu()
        for t, m, s in zip(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
            t.mul_(s).add_(m)
        return tensor.clamp_(0, 1)

    def visualize(self, dataset, split):
        fig, axes = plt.subplots(5, 4, figsize=(15, 10))
        axes = axes.flatten()
        fig.suptitle(f'{split.capitalize()} Split Samples', fontsize=20, y=1.01)

        for i, idx in enumerate(self.indices):
            if i >= len(axes):  # Prevent index error if <20 samples
                break
                
            img, label, _ = dataset[idx]
            
            # Handle tensor images
            if isinstance(img, torch.Tensor):
                img = self.denormalize(img)
                img_np = img.permute(1, 2, 0).numpy()
                img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            else:
                img_pil = img.copy()

            axes[i].imshow(img_pil)
            axes[i].set_title(f"GT -> {'Abnormal' if label else 'Normal'}")
            axes[i].axis('off')

        # Hide empty subplots if sample count < 20
        for j in range(i+1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()        
        plt.savefig(f"{self.save_dir}/{self.data_name}_{self.run_name}_{split}_data_vis.png")

    def data_analysis(self, cls_counts, save_name, color):
        print(f"\n{save_name.upper()} data analysis is in process...\n")
        width, text_width = 0.7, 0.05
        text_height = 0.5 if save_name == "train" else 0.1
        cls_names = list(cls_counts.keys())
        counts = list(cls_counts.values())        
        _, ax = plt.subplots(figsize=(20, 10))
        indices = np.arange(len(counts))
        ax.bar(indices, counts, width, color=color)
        ax.set_xlabel("Class Names", color="black")        
        ax.set(xticks=indices, xticklabels=cls_names)
        ax.set_xticklabels(cls_names, rotation = 90)
        ax.set_ylabel("Data Counts", color="black")
        ax.set_title("Dataset Class Imbalance Analysis")
        for i, v in enumerate(counts):
            ax.text(i - text_width, v + text_height, str(v), color="royalblue")
        plt.savefig(f"{self.save_dir}/{self.data_name}_{self.run_name}_{save_name}_data_analysis.png")
    
    def plot_pie_chart(self, cls_counts, save_name):
        print(f"{save_name.upper()} data pie chart visualization in process...\n")
        labels = list(cls_counts.keys())
        sizes = list(cls_counts.values())
        explode = [0.1] * len(labels)  # To highlight all slices equally (optional)
        
        plt.figure(figsize=(8, 8))
        plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.tab20.colors)
        plt.title("Class Distribution")
        plt.axis("equal")  # Equal aspect ratio ensures the pie chart is circular
        plt.savefig(f"{self.save_dir}/{self.data_name}_{self.run_name}_{save_name}_pie_chart.png")
    
    def run(self):

        for idx, split in enumerate(self.splits):
            dataset = self.datasets[split]
            class_counts = dataset.class_counts
            self.available_samples = min(len(dataset), self.num_samples)
            self.indices = random.sample(range(len(dataset)), self.available_samples)
            self.visualize(dataset=dataset, split=split)   
            self.data_analysis(cls_counts=class_counts, save_name=split, color=self.colors[idx])
            self.plot_pie_chart(cls_counts=class_counts, save_name=split)
        
    