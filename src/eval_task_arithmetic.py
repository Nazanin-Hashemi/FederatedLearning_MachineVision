import torch
import copy
import os
import sys

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import DinoClassifier
from src.datamodule import CIFAR100DataModule

def apply_task_arithmetic(base_model, federated_model, alpha=0.5):
    """
    Implements: W_new = W_base + alpha * (W_fed - W_base)
    Calculates the 'Task Vector' and scales it by alpha.
    """
    with torch.no_grad():
        base_sd = base_model.state_dict()
        fed_sd = federated_model.state_dict()
        new_state_dict = {}
        
        for name in base_sd:
            if name in fed_sd:
                # Calculate the Task Vector
                task_vector = fed_sd[name].to(base_sd[name].device) - base_sd[name]
                # Apply scaling and add back to base
                new_state_dict[name] = base_sd[name] + alpha * task_vector
            else:
                new_state_dict[name] = base_sd[name]
                
    base_model.load_state_dict(new_state_dict)
    return base_model

def evaluate(model, dataloader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Setup Data - CRITICAL: MATCH TRAINING RESOLUTION
    dm = CIFAR100DataModule(
        data_dir='./data', 
        batch_size=32,   # Lowered to avoid OOM with 224 resolution
        image_size=224   # MUST match your new datamodule.py setup
    )
    dm.setup("test")
    test_loader = dm.test_dataloader()

    # 2. Define Correct Paths
    base_ckpt_path = "checkpoints/dino-cifar100-baseline.ckpt"
    fed_ckpt_path = "results/checkpoints/last.ckpt" 

    if not os.path.exists(base_ckpt_path) or not os.path.exists(fed_ckpt_path):
        print("Checkpoints missing. Ensure paths are correct.")
        sys.exit(1)

    print(f"Loading Baseline from: {base_ckpt_path}")
    base_model = DinoClassifier.load_from_checkpoint(base_ckpt_path)
    
    print(f"Loading Federated Model from: {fed_ckpt_path}")
    fed_model = copy.deepcopy(base_model)
    
    checkpoint = torch.load(fed_ckpt_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Clean state dict keys
    cleaned_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    fed_model.load_state_dict(cleaned_state_dict, strict=False)

    # 3. Perform Alpha Sweep
    
    print(f"\n{'Alpha (Scaling Factor)':<25} | {'Test Accuracy':<15}")
    print("-" * 45)

    for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
        # Reset to base model to avoid cumulative alpha application
        temp_base = copy.deepcopy(base_model)
        arith_model = apply_task_arithmetic(temp_base, fed_model, alpha=alpha)
        
        acc = evaluate(arith_model, test_loader, device)
        star = "*" if alpha == 0.4 else "" 
        print(f"{alpha:<25.1f} | {acc*100:>13.2f}% {star}")