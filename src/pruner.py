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
        sparsity_level: Fraction of parameters to freeze (mask=0), for every strategy.
                       E.g., 0.7 means 70% frozen (30% updated).
        strategy: Masking strategy ("least_sensitive", "most_sensitive", etc.)
        seed: Seed for the "random" strategy.
    """
    def __init__(self, sparsity_level: float = 0.5, strategy: str = "least_sensitive", seed: int = 0):
        self.sparsity_level = sparsity_level
        self.strategy = strategy
        self.seed = seed
        self.global_mask = {}

    def _build_mask(self, scores: dict, train_lowest: bool) -> dict:
        """
        Train exactly round((1 - sparsity_level) * N) weights, picked globally by score.

        Shared by all strategies so that sparsity_level means the same thing everywhere [15].
        torch.topk selects an exact count, so ties cannot change the trained fraction.
        """
        all_scores = torch.cat([s.flatten() for s in scores.values()])
        n_train = round((1 - self.sparsity_level) * all_scores.numel())
        flat_mask = torch.zeros_like(all_scores)
        if n_train > 0:
            idx = torch.topk(all_scores, n_train, largest=not train_lowest).indices
            flat_mask[idx] = 1.0

        offset = 0
        for name, s in scores.items():
            self.global_mask[name] = flat_mask[offset:offset + s.numel()].view_as(s).cpu()
            offset += s.numel()
        return self.global_mask

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
        - Keeps exactly (1 - sparsity_level) of the weights trainable, picked by the strategy's score
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
        6. Mask: _build_mask trains the (1 - sparsity_level) fraction with the lowest
           (least_sensitive) or highest (most_sensitive) scores and freezes the rest
        
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

            # least_sensitive: update the lowest Fisher scores, freeze the rest
            # most_sensitive: update the highest Fisher scores, freeze the rest
            return self._build_mask(
                accumulated_fisher, train_lowest=(self.strategy == "least_sensitive")
            )
    

    def _random_mask(self, model):
        # Uniform random scores from a seeded generator, so masks are reproducible
        generator = torch.Generator().manual_seed(self.seed)
        scores = {
            name: torch.rand(p.shape, generator=generator)
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        return self._build_mask(scores, train_lowest=True)

    def _magnitude_mask(self, model):
        scores = {
            name: p.data.abs()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        return self._build_mask(scores, train_lowest=(self.strategy == "lowest_magnitude"))
