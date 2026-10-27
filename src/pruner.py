"""
Fisher Information-based Gradient Mask Calibration for Sparse Fine-tuning

This module implements the gradient mask calibration process as described in the task arithmetic
literature (e.g., [15], Sec. 4.2). The calibration identifies least-sensitive parameters through
multiple rounds of Fisher Information matrix diagonal estimation.

References:
    - Task Arithmetic methodology for sparse fine-tuning
    - Fisher Information matrix diagonal elements = squared gradients (∇²)
    - Multi-round calibration for robustness against gradient noise
"""
import torch
import torch.nn as nn
from tqdm import tqdm

class FisherPruner:
    """
    Calibrates gradient masks using Fisher Information matrix diagonal estimation.
    
    The calibration process:
    1. Performs multiple forward/backward passes without weight updates
    2. Accumulates squared gradients (Fisher Information diagonal) across rounds
    3. Averages scores to identify least-sensitive parameters
    4. Creates binary mask: mask=1 for parameters to update, mask=0 to freeze
    
    Args:
        sparsity_level: Fraction of parameters to freeze (mask=0). 
                       E.g., 0.7 means 70% frozen (30% updated).
        strategy: Masking strategy ("least_sensitive", "most_sensitive", etc.)
    """
    def __init__(self, sparsity_level: float = 0.5, strategy: str = "least_sensitive"):
        self.sparsity_level = sparsity_level
        self.strategy = strategy
        self.global_mask = {}

    def compute_mask(
        self, 
        model: nn.Module, 
        dataloader, 
        device: torch.device, 
        num_rounds: int = 3
    ) -> dict:
        """
        Compute gradient mask by identifying least-sensitive parameters.
        
        This implements Step 1 of sparse fine-tuning:
        - Calibrates mask in multiple rounds (see [15], Sec. 4.2 for rationale)
        - Identifies parameters with sensitivity scores below threshold
        - Returns binary mask: mask[name][param] = 1 (update) or 0 (freeze)
        
        Args:
            model: Model to calibrate mask for
            dataloader: Data loader for calibration (typically validation set)
            device: Computation device
            num_rounds: Number of calibration rounds for robustness
        
        Returns:
            Dictionary mapping parameter names to binary mask tensors
        """
        model.to(device)
        
        if self.strategy in ["least_sensitive", "most_sensitive"]:
            return self._fisher_mask(model, dataloader, device, num_rounds)
        elif self.strategy in ["lowest_magnitude", "highest_magnitude"]:
            return self._magnitude_mask(model)
        elif self.strategy == "random":
            return self._random_mask(model)
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}. Must be one of: "
                           f"least_sensitive, most_sensitive, lowest_magnitude, "
                           f"highest_magnitude, random")
        

    def _fisher_mask(self, model, dataloader, device, num_rounds):
        """
        Compute Fisher Information-based sensitivity scores through multiple calibration rounds.
        
        Process per round:
        1. Forward pass: Compute loss (no weight updates)
        2. Backward pass: Compute gradients
        3. Accumulate: Add squared gradients (Fisher Information diagonal = ∇²) to running sum
        4. Average: Normalize by number of samples in round
        
        After all rounds:
        5. Average: Compute mean sensitivity scores across rounds
        6. Threshold: Identify least-sensitive parameters (scores < threshold)
        7. Mask: Create binary mask (1=update least-sensitive, 0=freeze most-sensitive)
        
        Multi-round rationale: Averages out gradient noise and outliers for robust mask
        (see [15], Sec. 4.2 for detailed explanation).
        """
        model.eval()  # Set to eval mode (no batch norm updates)
        # Initialize Fisher Information accumulation tensors
        accumulated_fisher = {
            name: torch.zeros_like(p) 
            for name, p in model.named_parameters() 
            if p.requires_grad
        }
        criterion = nn.CrossEntropyLoss()
        
        # Multi-round calibration for robustness (see [15], Sec. 4.2)
        for r in range(num_rounds):
            torch.set_grad_enabled(True)  # Enable gradients for backward pass
            current_round_fisher = {
                name: torch.zeros_like(p) 
                for name, p in model.named_parameters() 
                if p.requires_grad
            }
            total_samples = 0
            
            # Accumulate squared gradients over entire dataloader
            for inputs, targets in tqdm(dataloader, desc=f"Fisher Round {r+1}/{num_rounds}"):
                inputs, targets = inputs.to(device), targets.to(device)
                total_samples += inputs.size(0)
                model.zero_grad()

                # Forward pass: Compute loss (weights unchanged)
                loss = criterion(model(inputs), targets)
                # Backward pass: Compute gradients
                loss.backward()

                # Accumulate squared gradients (Fisher Information diagonal elements)
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if name in current_round_fisher and p.grad is not None:
                            # Fisher Information: F_ii = (∂L/∂θ_i)²
                            # Weight by batch size for proper per-sample averaging
                            current_round_fisher[name] += p.grad.data.pow(2) * inputs.size(0)

            # Average Fisher scores over samples in this round
            with torch.no_grad():
                for name in accumulated_fisher:
                    # Normalize by total samples in round (per-sample average)
                    accumulated_fisher[name] += (current_round_fisher[name] / total_samples)
        
        # Average Fisher scores across all calibration rounds
        with torch.no_grad():
            for name in accumulated_fisher:
                accumulated_fisher[name] /= num_rounds

            # Compute global threshold based on sparsity level
            # Flatten all sensitivity scores into single tensor
            all_scores = torch.cat([f.view(-1) for f in accumulated_fisher.values()])
            # k: number of parameters to freeze (mask=0)
            k = int(all_scores.numel() * self.sparsity_level)

            if self.strategy == "least_sensitive":
                # Find k-th smallest score (threshold for least-sensitive parameters)
                threshold = torch.kthvalue(all_scores, k).values
                # mask=1 for parameters with score <= threshold (least sensitive → will update)
                # mask=0 for parameters with score > threshold (most sensitive → will freeze)
                for name, f in accumulated_fisher.items():
                    self.global_mask[name] = (f <= threshold).float().cpu()
            else:  # most_sensitive
                # Find k-th largest score (threshold for most-sensitive parameters)
                threshold = torch.kthvalue(all_scores, all_scores.numel() - k + 1).values
                # mask=1 for parameters with score >= threshold (most sensitive → will update)
                # mask=0 for parameters with score < threshold (least sensitive → will freeze)
                for name, f in accumulated_fisher.items():
                    self.global_mask[name] = (f >= threshold).float().cpu()
        return self.global_mask
    

    def _random_mask(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                rand = torch.rand_like(p)
                self.global_mask[name] = (rand > self.sparsity_level).float().cpu()
        return self.global_mask
    
    def _magnitude_mask(self, model):
        all_weights = []
        param_names = []
        for name, p in model.named_parameters():
            if p.requires_grad:
                all_weights.append(p.data.abs().view(-1))
                param_names.append(name)
        
        all_weights = torch.cat(all_weights)
        k = int(self.sparsity_level * all_weights.numel())
        threshold, _ = torch.kthvalue(all_weights, k)
        
        for name, p in model.named_parameters():
            if p.requires_grad:
                if self.strategy == "lowest_magnitude":
                    self.global_mask[name] = (p.data.abs() <= threshold).float().cpu()
                else:
                    self.global_mask[name] = (p.data.abs() > threshold).float().cpu()
        
        return self.global_mask
