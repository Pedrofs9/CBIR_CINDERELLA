
# Imports
import argparse
import os
import numpy as np
import datetime
import random
import json
import shutil
import pandas as pd
import datetime
from PIL import Image
# PyTorch Imports
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import TripletMarginLoss
import torch.nn as nn  
# Project Imports
from utilities_preproc import sample_manager
from utilities_traintest import TripletDataset, train_model, eval_model
from utilities_imgmodels import MODELS_DICT as models_dict, ModelEnsemble
from utilities_traintest import visualize_all_queries 
from xai_utils.xai_visualization import visualize_rankings_with_xai
# WandB Imports
import wandb

torch.backends.cudnn.benchmark = True  # Optimizes CUDA operations
torch.backends.cuda.enable_flash_sdp(False)  # Disables flash attention if present

# Configure memory settings before any CUDA operations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.9'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Disable for better performance


def set_seed(seed=10):
    """Set seed for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return

if __name__ == "__main__":
    # CLI Arguments
    parser = argparse.ArgumentParser(description='Cinderella BreLoAI Retrieval: Model Training with image data.')
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID to use")
    parser.add_argument('--config_json', type=str, default="config/config_image.json", help="JSON config file")
    parser.add_argument('--pickles_path', type=str, required=True, help="Path to pickle files")
    parser.add_argument('--results_path', type=str, help="Path to save results")
    parser.add_argument('--train_or_test', type=str, choices=["train", "test"], default="train", 
                       help="Execution mode: train or test")
    parser.add_argument('--checkpoint_path', type=str, help="Path to model checkpoints")
    parser.add_argument('--verbose', action='store_true', help="Verbose output")
    parser.add_argument('--ensemble_config', type=str, help="JSON file for ensemble configuration")
    parser.add_argument('--visualize', action='store_true', help="Enable ranking visualizations")
    parser.add_argument('--visualizations_path', type=str, help="Path to save ranking visualizations")
    parser.add_argument('--visualize_all', action='store_true', 
                   help="Generate visualizations for all query images")
    parser.add_argument('--max_visualizations', type=int, default=20,
                   help="Maximum number of visualizations to generate")    
    parser.add_argument('--visualize_triplets', action='store_true', 
                   help="Generate visualizations for triplets")    
    parser.add_argument('--generate_xai', action='store_true', help='Generate explanation maps')
    parser.add_argument('--xai_backend', choices=['Captum','MONAI'], help='XAI library to use')
    parser.add_argument('--xai_method', help='Specific explanation method') 
    parser.add_argument('--xai_batch_size', type=int, default=1, help='Batch size for XAI computations')          
    args = parser.parse_args()

    # Load configuration
    if args.train_or_test == "train":
        if args.ensemble_config:
            with open(args.ensemble_config, 'r') as f:
                config_json = json.load(f)
            is_ensemble = True
        else:
            with open(args.config_json, 'r') as f:
                config_json = json.load(f)
            is_ensemble = False
    else:  # test mode
        if args.ensemble_config:
            is_ensemble = True
            # For ensemble testing, load from checkpoint path
            with open(os.path.join(args.checkpoint_path, 'config.json'), 'r') as f:
                config_json = json.load(f)
        else:
            is_ensemble = False
            # Original single model test case
            with open(os.path.join(args.checkpoint_path, 'config.json'), 'r') as f:
                config_json = json.load(f)

    # Setup paths and device
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    if args.train_or_test == "train":
        experiment_results_path = os.path.join(args.results_path, timestamp)
        path_save = os.path.join(experiment_results_path, 'bin')
        os.makedirs(experiment_results_path, exist_ok=True)
        os.makedirs(path_save, exist_ok=True)
        
        # Save config copy
        shutil.copyfile(
            src=args.ensemble_config if is_ensemble else args.config_json,
            dst=os.path.join(experiment_results_path, 'config.json')
        )
    else:
        if is_ensemble:
            # For ensemble testing, use the ensemble config directly
            path_save = os.path.join(args.results_path, 'ensemble_results', timestamp)
            os.makedirs(path_save, exist_ok=True)
        else:
            # Original single model test case
            path_save = os.path.join(args.checkpoint_path, 'bin')
            with open(os.path.join(args.checkpoint_path, 'config.json'), 'r') as j:
                config_json = json.load(j)
    # Set seed
    set_seed(seed=config_json["seed"])

    # Initialize WandB if training
    if args.train_or_test == "train":
        wandb_project_config = {
            "seed": config_json["seed"],
            "lr": config_json.get("lr", 0.0001),
            "num_epochs": config_json["num_epochs"],
            "batch_size": config_json["batch_size"],
            "margin": config_json["margin"],
            "split_ratio": config_json["split_ratio"],
            "catalogue_type": config_json["catalogue_type"],
            "doctor_code": config_json["doctor_code"],
            "model_name": config_json["model_name"],
            "fusion_type": config_json.get("fusion_type", "projection"),
            "transformer_dim": config_json.get("transformer_dim", 512),
            "nhead": config_json.get("nhead", 4)
        }
        wandb_run = wandb.init(
            project="bcs-aesth-mm-attention-mir",
            name=config_json["model_name"]+'_'+timestamp,
            config=wandb_project_config
        )
    else:
        wandb_run = None

    # Load data
    QNS_list_image_train, QNS_list_image_test, _, _ = sample_manager(pickles_path=args.pickles_path)

    checkpoint_paths = config_json.get("checkpoint_paths", None)
    if checkpoint_paths and not torch.cuda.is_available():
        # Create a wrapper that loads on CPU
        def safe_load(path):
            return torch.load(path, map_location='cpu')
    else:
        def safe_load(path):
            return torch.load(path)


if is_ensemble:
    model = ModelEnsemble(
        model_names=config_json["model_names"],
        checkpoint_paths=checkpoint_paths,
        trainable=config_json.get("trainable", False),
        models_dict=models_dict,
        fusion_type=config_json.get("fusion_type", "projection"), 
        mlp_dims=config_json.get("mlp_dims", None),
        proj_dim=config_json.get("proj_dim", None),
        use_l2_norm=config_json.get("use_l2_norm", True),
        transformer_dim=config_json.get("transformer_dim", 512), 
        nhead=config_json.get("nhead", 4),                     
        num_layers=config_json.get("num_layers", 1),
        load_function=safe_load
    )
else:
    model = models_dict[config_json["model_name"]]

    if args.train_or_test == "test":
        model_path = os.path.join(path_save, "model_final.pt")
        try:
            # Load model with memory optimization
            state_dict = torch.load(model_path, map_location='cpu')

            # Handle 'base_model.' prefix if present
            if any(k.startswith("base_model.") for k in state_dict.keys()):
                state_dict = {k.replace("base_model.", ""): v for k, v in state_dict.items()}

            model.load_state_dict(state_dict, strict=False)
            del state_dict
        except Exception as e:
            print(f"Error loading model: {e}")
            raise


    # Move model to device with memory check
    if device.type == 'cuda':
        # More detailed memory check
        total_mem = torch.cuda.get_device_properties(device).total_memory
        free_mem = total_mem - torch.cuda.memory_allocated(device)
        model_mem = sum(p.numel() * p.element_size() for p in model.parameters())
        xai_buffer = 2 * 1024**3  # 2GB buffer for XAI computations
        
        if model_mem > (free_mem - xai_buffer):
            raise RuntimeError(f"Insufficient GPU memory. Required: {model_mem/1024**3:.2f}GB, Available: {free_mem/1024**3:.2f}GB")
    
    model = model.to(device)

    # Create dataloaders FIRST (before any similarity checks)
    transform = model.get_transform()
    train_dataset = TripletDataset(QNS_list=QNS_list_image_train, transform=transform)
    test_dataset = TripletDataset(QNS_list=QNS_list_image_test, transform=transform)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config_json["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config_json["batch_size"],
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # Debug checks AFTER dataloader creation
    if args.verbose and isinstance(model, ModelEnsemble):
        print("\n=== Running Ensemble Diagnostics ===")
        
        # 1. Feature similarity check
        print("\n[1/2] Model Feature Similarity:")
        debug_dataset = TripletDataset(QNS_list=QNS_list_image_test[:1], transform=transform)
        debug_loader = DataLoader(debug_dataset, batch_size=1)
        sample = next(iter(debug_loader))['query'].to(device)
        
        with torch.no_grad():
            features = [m(sample) for m in model.models]
            for i in range(len(features)):
                for j in range(i, len(features)):
                    sim = F.cosine_similarity(features[i], features[j]).mean().item()
                    print(f"  {config_json['model_names'][i]} vs {config_json['model_names'][j]}: {sim:.4f}")

        # 2. Individual model performance (only in test mode)
        if args.train_or_test == "test":
            for i, name in enumerate(config_json["model_names"]):
                try:
                    single_model = models_dict[name]().to(device)
                    if config_json["checkpoint_paths"] and i < len(config_json["checkpoint_paths"]):
                        single_model.load_state_dict(
                            torch.load(config_json["checkpoint_paths"][i],
                                    map_location=device),
                            strict=False
                        )
                    
                    _, test_acc = eval_model(single_model, test_loader, QNS_list_image_test, device)
                    print(f"{name}: Test Acc = {test_acc:.4f}")
                    del single_model
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"⚠️ Failed to test {name}: {str(e)}")
                    
    # Setup training
    criterion = TripletMarginLoss(margin=config_json["margin"], p=2)
            
    if args.train_or_test == "train":
        # Initialize training variables
        best_model = model
        final_epoch_loss = float('inf')
        save_success = False
        
        def safe_save_model(model, path):
            """Helper function for robust model saving"""
            try:
                # First try the intended path
                torch.save(model.state_dict(), path)
                return True
            except Exception as e:
                print(f"Failed to save model to {path}: {str(e)}")
                try:
                    # Fallback to temporary location
                    temp_path = f"/tmp/{os.path.basename(path)}"
                    torch.save(model.state_dict(), temp_path)
                    print(f"Model saved to temporary location: {temp_path}")
                    return True
                except Exception as e:
                    print(f"Failed to save model to temporary location: {str(e)}")
                    return False


        if is_ensemble:
            # Ensemble-specific logic
            if config_json.get("trainable", False) or config_json.get("fusion_type") in ["mlp", "transformer"]:
                if config_json.get("use_mlp", False):  # If use_mlp is not true, returns false and does not execute this section 
                    ##############################################
                    # Two-Phase Training for MLP-based Ensembles #
                    ##############################################
                    
                    # --- Phase 1: Train only MLP (freeze base models) ---
                    phase1_epochs = max(1, int(config_json["num_epochs"] * 0.3))
                    print(f"\n=== PHASE 1: Training MLP only ({phase1_epochs}/{config_json['num_epochs']} epochs) ===")
                    
                    # Freeze all individual models
                    for m in model.models:
                        for param in m.parameters():
                            param.requires_grad = False
                    
                    # Train only MLP parameters
                    optimizer = optim.Adam(
                        model.mlp.parameters(),
                        lr=config_json["lr"]
                    )
                    
                    try:
                        model, phase1_loss, _ = train_model(
                            model=model,
                            train_loader=train_loader,
                            test_loader=test_loader,
                            QNS_list_train=QNS_list_image_train,
                            QNS_list_test=QNS_list_image_test,
                            optimizer=optimizer,
                            criterion=criterion,
                            num_epochs=phase1_epochs,
                            device=device,
                            path_save=path_save,
                            wandb_run=wandb_run
                        )
                        
                        # --- Phase 2: Fine-tune entire ensemble ---
                        phase2_epochs = max(1, config_json["num_epochs"] - phase1_epochs)
                        print(f"\n=== PHASE 2: Fine-tuning full ensemble ({phase2_epochs}/{config_json['num_epochs']} epochs) ===")
                        
                        # Unfreeze all models
                        for m in model.models:
                            for param in m.parameters():
                                param.requires_grad = True
                        
                        # Create optimizer with potentially different learning rates
                        if "model_lrs" in config_json:
                            optim_params = []
                            for i, m in enumerate(model.models):
                                optim_params.append({
                                    "params": m.parameters(),
                                    "lr": config_json["model_lrs"][i]
                                })
                            optim_params.append({
                                "params": model.mlp.parameters(),
                                "lr": config_json["lr"]
                            })
                            optimizer = optim.Adam(optim_params)
                        else:
                            optimizer = optim.Adam(model.parameters(), lr=config_json["lr"])
                        
                        best_model, final_epoch_loss, _ = train_model(
                            model=model,
                            train_loader=train_loader,
                            test_loader=test_loader,
                            QNS_list_train=QNS_list_image_train,
                            QNS_list_test=QNS_list_image_test,
                            optimizer=optimizer,
                            criterion=criterion,
                            num_epochs=phase2_epochs,
                            device=device,
                            path_save=path_save,
                            wandb_run=wandb_run
                        )
                        
                        # Save final model
                        save_success = safe_save_model(best_model, os.path.join(path_save, "model_final.pt"))
                        
                    except Exception as e:
                        print(f"Training failed: {str(e)}")
                        # Try to save current state before exiting
                        save_success = safe_save_model(model, os.path.join(path_save, "model_interrupted.pt"))
                        raise
                    
                else:
                    ####################################
                    # Simple Concatenation (No MLP) #
                    ####################################
                    print(f"\n=== Training Simple Concatenation Ensemble ({config_json['num_epochs']} epochs) ===")
                    optimizer = optim.Adam(model.parameters(), lr=config_json["lr"])
                    
                    try:
                        best_model, final_epoch_loss, _ = train_model(
                            model=model,
                            train_loader=train_loader,
                            test_loader=test_loader,
                            QNS_list_train=QNS_list_image_train,
                            QNS_list_test=QNS_list_image_test,
                            optimizer=optimizer,
                            criterion=criterion,
                            num_epochs=config_json["num_epochs"],
                            device=device,
                            path_save=path_save,
                            wandb_run=wandb_run
                        )
                        save_success = safe_save_model(best_model, os.path.join(path_save, "model_final.pt"))
                    except Exception as e:
                        print(f"Training failed: {str(e)}")
                        save_success = safe_save_model(model, os.path.join(path_save, "model_interrupted.pt"))
                        raise
            else:

                # Ensemble with frozen base models but trainable fusion
                print(f"\n=== Training Ensemble Fusion Only ({config_json['num_epochs']} epochs) ===")
                
                # Freeze all individual models
                for m in model.models:
                    for param in m.parameters():
                        param.requires_grad = False
                
                # Only train fusion parameters
                trainable_params = []
                if hasattr(model, 'fusion'):
                    trainable_params.extend(model.fusion.parameters())
                if hasattr(model, 'feature_extractor'):
                    trainable_params.extend(model.feature_extractor.parameters())
                if hasattr(model, 'transformer_decoder'):
                    trainable_params.extend(model.transformer_decoder.parameters())
                if hasattr(model, 'fc_out'):
                    trainable_params.extend(model.fc_out.parameters())
                
                optimizer = optim.Adam(trainable_params, lr=config_json["lr"])
                
                try:
                    best_model, final_epoch_loss, _ = train_model(
                        model=model,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        QNS_list_train=QNS_list_image_train,
                        QNS_list_test=QNS_list_image_test,
                        optimizer=optimizer,
                        criterion=criterion,
                        num_epochs=config_json["num_epochs"],
                        device=device,
                        path_save=path_save,
                        wandb_run=wandb_run
                    )
                    save_success = safe_save_model(best_model, os.path.join(path_save, "model_final.pt"))
                except Exception as e:
                    print(f"Training failed: {str(e)}")
                    save_success = safe_save_model(model, os.path.join(path_save, "model_interrupted.pt"))
                    raise
        else:
            ##########################
            # Single Model Training #
            ##########################
            print(f"\n=== Training Single Model ({config_json['num_epochs']} epochs) ===")
            optimizer = optim.Adam(model.parameters(), lr=config_json["lr"])
            
            try:
                best_model, final_epoch_loss, _ = train_model(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    QNS_list_train=QNS_list_image_train,
                    QNS_list_test=QNS_list_image_test,
                    optimizer=optimizer,
                    criterion=criterion,
                    num_epochs=config_json["num_epochs"],
                    device=device,
                    path_save=path_save,
                    wandb_run=wandb_run
                )
                save_success = safe_save_model(best_model, os.path.join(path_save, "model_final.pt"))
            except Exception as e:
                print(f"Training failed: {str(e)}")
                save_success = safe_save_model(model, os.path.join(path_save, "model_interrupted.pt"))
                raise
        
        if save_success:
            print("Model saved successfully")
        
        if wandb_run:
            wandb_run.finish()

    else:  # Evaluation mode
        if args.visualize_all:
            QNS_lists = {
                #'train': QNS_list_image_train,
                'test': QNS_list_image_test
            }
            
            # Memory optimization wrapper
            def safe_visualize():
                try:
                    # Clear cache before starting
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                    
                    visualize_all_queries(
                        model=model,
                        QNS_lists=QNS_lists,
                        transform=model.get_transform(),
                        device=device,
                        output_dir=args.visualizations_path,
                        max_visualizations=min(args.max_visualizations, 500),
                        xai_method=args.xai_method if (args.generate_xai and args.xai_method) else None,
                        xai_backend=args.xai_backend if args.generate_xai else None  
                    )
                except RuntimeError as e:
                    print(f"Visualization failed: {e}")
                    if "CUDA out of memory" in str(e):
                        print("Trying again with reduced batch size...")
                        visualize_all_queries(
                            model=model,
                            QNS_lists=QNS_lists,
                            transform=model.get_transform(),
                            device=torch.device('cpu'),  # Fallback to CPU
                            output_dir=args.visualizations_path,
                            max_visualizations=min(args.max_visualizations, 200),  # Further reduced
                        )
            
            safe_visualize()
            
        else:
            # Determine if we should visualize queries (only if --visualize is set and not --visualize_triplets)
            visualize_queries = args.visualize and not args.visualize_triplets
            
            # And in the eval_model calls:
            train_acc, train_ndcg = eval_model(
                model=model,
                eval_loader=train_loader,
                QNS_list_eval=QNS_list_image_train if visualize_queries else None,
                device=device,
                visualize=args.visualize_triplets,
                output_dir=args.visualizations_path,
                xai_method=args.xai_method if args.generate_xai else None,
                xai_backend=args.xai_backend if args.generate_xai else None
            )
            test_acc, test_ndcg = eval_model(
                model=model,
                eval_loader=test_loader,
                QNS_list_eval=QNS_list_image_test if visualize_queries else None,
                device=device,
                visualize=args.visualize_triplets,
                output_dir=args.visualizations_path,
                xai_backend=args.xai_backend if args.generate_xai else None  # Add this
            )
            
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # XAI Generation
            if args.generate_xai and args.train_or_test == "test":
                generate_xai_maps( 
                    model=model,
                    device=device,
                    eval_dataloader=test_loader,
                    xai_backend=args.xai_backend,
                    xai_method=args.xai_method,
                    results_dir=args.results_path  
                    
                )

            if args.visualize:
                generate_xai_plots(
                    results_path=args.results_path 
                )

            results_dict = {
                "train_acc": [train_acc],
                "train_ndcg": [train_ndcg],
                "test_acc": [test_acc],
                "test_ndcg": [test_ndcg]
            }
            eval_df = pd.DataFrame.from_dict(results_dict)
            eval_df.to_csv(os.path.join(path_save, "eval_results.csv"))
