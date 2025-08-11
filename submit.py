import argparse
from dataset import CellDataModule
from transformations import get_fts
from inference import InferenceEnsemble

def parse_args():
 
    parser = argparse.ArgumentParser(description='Kaggle Competitions Image Classification Submission Pipeline')    

    parser.add_argument('--im_size', type=int, default=224, help='Image size')
    parser.add_argument('--bs', type=int, default=16, help='Batch size')
    parser.add_argument('--data_name', type=str, default='digit', help='Data name')
    parser.add_argument('--vis_dir', type=str, default='vis', help='Visualization directory')
    parser.add_argument('--project_name', type=str, default='kaggle', help='Project name')
    parser.add_argument('--run_name', type=str, default='triplet', help='Run name')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')            
    # parser.add_argument('--save_dir', type=str, default='/home/bekhzod/Desktop/backup/kaggle/saved_models', help='Directory to save models')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save models')
    parser.add_argument('--embeddings_dir', type=str, default='/home/bekhzod/Desktop/backup/kaggle/embeddings_224', help='Directory to save embeddings')    
    # parser.add_argument('--model_names', nargs='+', default=["ecaresnet101d", "resnet101", "resnext101_32x8d", "rexnet_200", "vit_large_patch16_224", "swin_large_patch4_window7_224"], help='List of model names')    
    # parser.add_argument('--model_names', nargs='+', default=["ecaresnet101d", "resnet101", "resnext101_32x8d", "rexnet_200", "vit_large_patch16_224", "convnext_large",  "coatnet_0_rw_224.sw_in1k", "eva02_base_patch14_224"], help='List of model names')    
    # parser.add_argument('--model_names', nargs='+', default=["ecaresnet101d", "resnet101", "resnext101_32x8d", "rexnet_200", "vit_large_patch16_224", "swin_large_patch4_window7_224", "convnext_large", "coatnet_0_rw_224.sw_in1k", "eva02_base_patch14_224"], help='List of model names')    
    # parser.add_argument('--model_names', nargs='+', default=["ecaresnet269d", "resnet152d", "resnext101_64x4d", "rexnet_300", "mobilenetv3_large_100",  "vit_base_patch16_224", "swinv2_cr_small_ns_224", "convnextv2_base", "swin_base_patch4_window7_224", "convnext_large",  "efficientnetv2_rw_m", "deit_base_patch16_224"], help='List of model names')    
    # parser.add_argument('--model_names', nargs='+', default=["ecaresnet269d", "resnet152d", "resnext101_64x4d", "rexnet_300", "mobilenetv3_large_100",  "vit_base_patch16_224", "swinv2_cr_small_ns_224", "convnextv2_base", "swin_base_patch4_window7_224", "convnext_large"], help='List of model names')    
    parser.add_argument('--model_names', nargs='+', default=["ecaresnet269d", "resnet152d", "resnext101_64x4d", "rexnet_300", "vit_base_patch16_224", "swinv2_cr_small_ns_224", "convnextv2_base", "mobilenetv3_large_100", "swin_base_patch4_window7_224", "convnext_large",  "efficientnetv2_rw_m", "deit_base_patch16_224"], help='List of model names')    
    # parser.add_argument('--model_names', nargs='+', default=["ecaresnet101d", "resnet101", "resnext101_32x8d", "rexnet_200", "vit_large_patch16_224", "swin_large_patch4_window7_224", "convnext_large",  "coatnet_0_rw_224.sw_in1k"], help='List of model names')
    # parser.add_argument('--model_names', nargs='+', default=["ecaresnet101d", "resnet101", "resnext101_32x8d", "rexnet_200", "vit_large_patch16_224", "swin_large_patch4_window7_224", "convnext_large"], help='List of model names')
    # parser.add_argument('--model_names', nargs='+', default=["ecaresnet269d", "resnet152d", "resnext101_64x4d", "rexnet_300", "vit_base_patch16_224", "swinv2_cr_small_ns_224", "convnext_large"], help='List of model names')    
    # parser.add_argument('--model_names', nargs='+', default=["ecaresnet50d", "resnet50", "resnext50_32x4d"], help='List of model names')    
    # parser.add_argument('--model_names', nargs='+', default=["ecaresnet101d", "resnet101", "resnext101_32x8d"], help='List of model names')    
    # parser.add_argument('--model_names', nargs='+', default=[ "swin_large_patch4_window7_224", "maxvit_large_tf_224", "convnext_large",  "coatnet_0_rw_224.sw_in1k", "eva_giant_patch14_224" ], help='List of model names')    
    # parser.add_argument('--model_names', nargs='+', default=["resnext101_32x8d", "rexnet_200", "vit_large_patch16_224"], help='List of model names')        
    # parser.add_argument('--model_names', nargs='+', default=["ecaresnet101d", "resnet101", "resnext101_32x8d", "rexnet_200"], help='List of model names')
    parser.add_argument('--n_ims', type=int, default=18, help='Number of images for visualization')
    parser.add_argument('--rows', type=int, default=6, help='Rows for visualization grid')    
    parser.add_argument('--num_ims', type=int, default=20, help='Number of images for ensemble inference')
    parser.add_argument('--infer_rows', type=int, default=4, help='Rows for ensemble inference visualization')    
    return parser.parse_args()

def main():
    args = parse_args()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tfs, eval_tfs = get_fts(im_size=args.im_size, mean=mean, std=std)

    dm = CellDataModule(
        data_name=args.data_name,
        batch_size=args.bs,
        train_transform=train_tfs,
        eval_transform=eval_tfs,
        num_workers=4,
        persistent_workers=True,
        )
    dm.setup()
    
    ts_dl = dm.test_dataloader()
    
    classes = dm.train_dataset.class_names
    
    print(f"There are {len(ts_dl)}  batches in test  dataloader.")
    print(f"Class names -> {classes}")

    ensemble_infer = InferenceEnsemble(
        model_names=args.model_names,
        device=args.device,        
        save_dir=args.save_dir,
        data_name=args.data_name,
        run_name=args.run_name,
        test_dl=ts_dl,
        embeddings_dir=args.embeddings_dir,
        im_size=args.im_size
    )
    ensemble_infer.run(
        num_ims=args.num_ims,
        rows=args.infer_rows,
        save_submission=True,
        submission_path=f'{args.data_name}_ensemble_submission_{len(args.model_names)}_models.csv')

if __name__ == '__main__':
    main()
