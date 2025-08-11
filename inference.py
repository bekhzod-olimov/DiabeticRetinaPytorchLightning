import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import os, torch
from model import CellClassifier
# from pytorch_grad_cam import GradCAM
from torchvision import transforms as T
# from pytorch_grad_cam.utils.image import show_cam_on_image

class InferenceEnsemble:
    def __init__(self, model_names, save_dir, device, test_dl, data_name, run_name, embeddings_dir=None, im_size=224):
        self.model_names = model_names        
        self.device = device        
        self.save_dir = save_dir
        self.data_name, self.run_name = data_name, run_name
        self.test_dl = test_dl        
        class_names_file_path = f'{embeddings_dir}/{data_name}_class_names.pkl'
        with open(class_names_file_path, 'rb') as handle: self.cls_names = pickle.load(handle)            
        print(f"New class names -> {self.cls_names}")        
        self.im_size = im_size
        self.load_models()

    @staticmethod
    def tensor_2_im(t, t_type="rgb"):
        
        gray_tfs = T.Compose([ T.Normalize(mean=[0.], std=[1/0.5]), T.Normalize(mean=[-0.5], std=[1]) ])
        rgb_tfs = T.Compose([ T.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]), T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]) ])
        invTrans = gray_tfs if t_type == "gray" else rgb_tfs
        return (invTrans(t) * 255).detach().squeeze().cpu().permute(1, 2, 0).numpy().astype(np.uint8)

    def get_best_ckpt(self, ckpt_list):
        
        ckpt_list = [ckpt for ckpt in ckpt_list if self.run_name in ckpt]
        best_acc = -float('inf')
        best_ckpt = None
        
        for ckpt in ckpt_list:            
            # Extract the val_acc value from the filename
            import re
            match = re.search(r"val_acc=(\d+\.\d+)", ckpt)
            if match:
                acc = float(match.group(1))
                if acc > best_acc:
                    best_acc = acc
                    best_ckpt = ckpt
                    
        return best_ckpt
    
    def load_models(self):

        self.models = []
        for model_name in self.model_names:
            model = self.model = CellClassifier(
                model_name=model_name, 
                run_name=self.run_name,
                class_counts=None,
                num_classes=len(self.cls_names),            
            )            
            from glob import glob
            ckpt_name = self.get_best_ckpt(glob(f"{os.path.join(self.save_dir, self.data_name, model_name)}/*.ckpt"))
            print(ckpt_name)
            model.load_state_dict(torch.load(ckpt_name, weights_only=True, map_location="cpu")["state_dict"])
            # model.load_state_dict(torch.load(f"{self.save_dir}/{model_name}/{self.data_name}_{self.run_name}_{model_name}_best_model.pth", weights_only=True, map_location="cpu"))
            self.models.append(model.eval().to(self.device))
    
    def save_submission(self, im_paths, preds, submission_path):
        
        idx_to_cls = {v: k for k, v in self.cls_names.items()} if isinstance(self.cls_names, dict) else {i: name for i, name in enumerate(self.cls_names)}        
        
        if self.data_name in ["food"]: submission_data = { "image_id": [os.path.basename(p) for p in im_paths], "label": [idx_to_cls[p] for p in preds] }

        elif self.data_name in ["art"]: submission_data = { "id": [os.path.splitext(os.path.basename(p))[0] for p in im_paths], "predict": [idx_to_cls[p] for p in preds] }

        elif self.data_name in ["kenya_food"]: submission_data = { "id": [os.path.splitext(os.path.basename(p))[0] for p in im_paths], "class": [idx_to_cls[p] for p in preds] }

        elif self.data_name == "retina": submission_data = { "Id": [os.path.basename(p) for p in im_paths], "Category": [idx_to_cls[p] for p in preds] }

        elif self.data_name == "diabetic_retina": submission_data = { "id_code": [os.path.splitext(os.path.basename(p))[0] for p in im_paths], "diagnosis": [idx_to_cls[p] for p in preds] }

        elif self.data_name == "digit": submission_data = { "ImageId": [(p+1) for p in im_paths], "Label": [idx_to_cls[p] for p in preds] }        

        submission_df = pd.DataFrame(submission_data)
        submission_df.to_csv(submission_path, index=False)
        print(f"Saved submission file to {submission_path}")

    def run(self, num_ims, rows, save_submission=False, submission_path="submission.csv"):
        preds, images, im_paths, logitss = [], [], [], []
        print(f"Running ensemble inference with {len(self.models)} models...")
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(self.test_dl)):
                im_path, im = batch["im_path"], batch["im"].to(self.device)
                # Get logits from all models and average them
                logits_list = [model(im)[0] for model in self.models]  # Unpack logits
                avg_logits = torch.mean(torch.stack(logits_list), dim=0)
                pred_class = torch.argmax(avg_logits, dim=1)
                images.append(im.cpu())
                logitss.append(avg_logits.cpu())
                preds.append(pred_class.item())

                if self.data_name in ["food", "art", "kenya_food", "retina", "diabetic_retina"]:
                    if isinstance(im_path, (list, tuple)):
                        im_path = im_path[0]
                    if hasattr(im_path, "item"):
                        im_path = im_path.item()
                    im_paths.append(im_path)
                elif self.data_name == "digit":
                    im_paths.append(idx)
                else:
                    print(f"Submission data is not implemented for {self.data_name}")

        if save_submission:
            self.save_submission(im_paths, preds, submission_path)
