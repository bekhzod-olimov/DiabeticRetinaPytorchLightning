import os, torch, random, numpy as np, matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms as T

class DatasetVisualizer:
    def __init__(self, data, sample_type, run_name, data_name, save_dir, num_samples=20):
        
        self.sample_type = sample_type
        self.data = data
        self.save_dir = save_dir
        self.run_name, self.data_name = run_name, data_name         
        self.num_samples = num_samples
        os.makedirs(self.save_dir, exist_ok = True)        
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
                
            img, label, bboxes = dataset[idx]
            
            # Handle tensor images
            if isinstance(img, torch.Tensor):
                img = self.denormalize(img)
                img_np = img.permute(1, 2, 0).numpy()
                img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            else:
                img_pil = img.copy()
                
            draw = ImageDraw.Draw(img_pil)
            W, H = img_pil.size

            # Convert bounding boxes
            if isinstance(bboxes, torch.Tensor):
                bboxes = bboxes.cpu().numpy()

            # Draw bounding boxes
            for bbox in (bboxes if len(bboxes.shape) == 2 else [bboxes]):
                if len(bbox) == 4 and not np.all(bbox == 0):
                    if np.max(bbox) <= 1.0:  # Normalized coordinates
                        cx, cy, w, h = bbox
                        x1 = (cx - w/2) * W
                        y1 = (cy - h/2) * H
                        x2 = (cx + w/2) * W
                        y2 = (cy + h/2) * H
                    else:  # Absolute coordinates
                        x1, y1, x2, y2 = bbox
                    
                    if x1 < 0: x1, y1, x2, y2 = 0, 0, 0, 0
                    draw.rectangle([x1, y1, x2, y2], outline='red', width=5)

            axes[i].imshow(img_pil)
            axes[i].set_title(f"GT -> {'Abnormal' if label else 'Normal'}")
            axes[i].axis('off')

        # Hide empty subplots if sample count < 20
        for j in range(i+1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()        
        plt.savefig(f"{self.save_dir}/{self.sample_type}_{self.data_name}_{self.run_name}_{split}_data_vis.png")

    def data_analysis(self, cls_counts, save_name, color):
        print(cls_counts)
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
        plt.savefig(f"{self.save_dir}/{self.sample_type}_{self.data_name}_{self.run_name}_{save_name}_data_analysis.png")
    
    def plot_pie_chart(self, cls_counts, save_name):
        print(f"{save_name.upper()} data pie chart visualization in process...\n")
        labels = list(cls_counts.keys())
        sizes = list(cls_counts.values())
        explode = [0.1] * len(labels)  # To highlight all slices equally (optional)
        
        plt.figure(figsize=(8, 8))
        plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.tab20.colors)
        plt.title("Class Distribution")
        plt.axis("equal")  # Equal aspect ratio ensures the pie chart is circular
        plt.savefig(f"{self.save_dir}/{self.sample_type}_{self.data_name}_{self.run_name}_{save_name}_pie_chart.png")
    
    def run(self):           
        
        self.available_samples = min(len(self.data), self.num_samples)
        self.indices = random.sample(range(len(self.data)), self.available_samples)
        self.visualize(dataset=self.data, split="all")   
        self.data_analysis(cls_counts=self.data.class_counts, save_name="all", color=self.colors[0])
        self.plot_pie_chart(cls_counts=self.data.class_counts, save_name="all")
        
    