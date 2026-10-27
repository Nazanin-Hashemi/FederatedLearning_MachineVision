"""
Fisher Information-based Gradient Mask Calibration for Sparse Fine-tuning

This module implements the gradient mask calibration process of TaLoS [15], Sec. 4.2:
the mask is calibrated in multiple rounds with a progressive sparsity schedule, re-scoring
the model with the current mask applied in every round.

References:
    [15] Iurada et al., Efficient Model Editing with Task-Localized Sparse Fine-tuning. ICLR 2025.
         Reference code: https://github.com/iurada/talos-task-arithmetic (vision/pruners.py,
         vision/prune_finetune.py)
"""
import torch
import torch.nn as nn
from tqdm import tqdm

# Scale applied to excluded weights while re-scoring, as in the TaLoS masked layers [15]
EXCLUDED_WEIGHT_SCALE = 0.1


class FisherPruner:
    """
    Calibrates gradient masks for sparse fine-tuning.

    The calibration process:
    1. Scores every scored weight (Fisher Information diagonal, |w| or random)
    2. Fisher strategies only: repeats scoring over several rounds, shrinking the trainable
       set each round and re-scoring with the current mask applied [15]
    3. Creates binary mask: mask=1 for parameters to update, mask=0 to freeze

    Parameters listed in dense_params are not scored and get no mask, so SparseSGDM
    updates them densely (TaLoS trains the positional and class embeddings densely [15]).
    Sparsity is counted over scored weights only.

    Args:
        sparsity_level: Fraction of scored weights to freeze (mask=0), for every strategy.
                       E.g., 0.7 means 70% frozen (30% updated).
        strategy: Masking strategy ("least_sensitive", "most_sensitive", etc.)
        seed: Seed for the "random" strategy.
        dense_params: Parameter names (last component) that are trained densely.
    """
    def __init__(
        self,
        sparsity_level: float = 0.5,
        strategy: str = "least_sensitive",
        seed: int = 0,
        dense_params: tuple = ("cls_token", "pos_embed"),
    ):
        self.sparsity_level = sparsity_level
        self.strategy = strategy
        self.seed = seed
        self.dense_params = dense_params
        self.global_mask = {}
        # Fraction of scored weights left trainable, after each round and at the end
        self.round_fractions = []
        self.trained_fraction = None

    def _scored_params(self, model: nn.Module) -> dict:
        return {
            name: p
            for name, p in model.named_parameters()
            if p.requires_grad and name.split(".")[-1] not in self.dense_params
        }

    def _build_mask(
        self,
        scores: dict,
        train_lowest: bool,
        keep_fraction: float = None,
        eligible: dict = None,
    ) -> dict:
        """
        Train exactly round(keep_fraction * N) weights, picked globally by score.

        Shared by all strategies so that sparsity_level means the same thing everywhere [15].
        torch.topk selects an exact count, so ties cannot change the trained fraction.

        Args:
            keep_fraction: Fraction of all N scored weights to train (default 1 - sparsity_level).
            eligible: Optional mask from the previous round; only weights with eligible=1 can
                      be picked, so excluded weights never come back (TaLoS global_copy [15]).
        """
        if keep_fraction is None:
            keep_fraction = 1 - self.sparsity_level
        all_scores = torch.cat([s.flatten().cpu() for s in scores.values()])
        n_train = round(keep_fraction * all_scores.numel())

        if eligible is not None:
            # Excluded weights get the worst possible score: +inf when keeping the lowest
            all_eligible = torch.cat([eligible[name].flatten() for name in scores])
            excluded_score = torch.inf if train_lowest else -torch.inf
            all_scores = torch.where(all_eligible == 1, all_scores, excluded_score)
            n_train = min(n_train, int(all_eligible.sum().item()))

        flat_mask = torch.zeros_like(all_scores)
        if n_train > 0:
            idx = torch.topk(all_scores, n_train, largest=not train_lowest).indices
            flat_mask[idx] = 1.0

        offset = 0
        for name, s in scores.items():
            self.global_mask[name] = flat_mask[offset:offset + s.numel()].view_as(s)
            offset += s.numel()
        self.trained_fraction = n_train / all_scores.numel()
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
        - Fisher strategies: calibrates the mask in num_rounds progressive rounds ([15], Sec. 4.2)
        - Magnitude and random strategies: one-shot, num_rounds is ignored
        - Keeps exactly (1 - sparsity_level) of the scored weights trainable
        - Returns binary mask: mask[name][param] = 1 (update) or 0 (freeze)

        The trained fraction is printed and stored in self.trained_fraction
        (per round in self.round_fractions), e.g. for experiment logging.

        Args:
            model: Model to calibrate mask for
            dataloader: Data loader for calibration (typically validation set)
            device: Computation device
            num_rounds: Number of calibration rounds (Fisher strategies only)

        Returns:
            Dictionary mapping scored parameter names to binary mask tensors
        """
        model.to(device)
        self.round_fractions = []

        if self.strategy in ["least_sensitive", "most_sensitive"]:
            mask = self._fisher_mask(model, dataloader, device, num_rounds)
        elif self.strategy in ["lowest_magnitude", "highest_magnitude"]:
            mask = self._magnitude_mask(model)
        elif self.strategy == "random":
            mask = self._random_mask(model)
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}. Must be one of: "
                           f"least_sensitive, most_sensitive, lowest_magnitude, "
                           f"highest_magnitude, random")

        n_scored = sum(m.numel() for m in mask.values())
        print(f"[pruner] {self.strategy}: trained fraction {self.trained_fraction:.4f} "
              f"of {n_scored} scored weights")
        return mask


    def _fisher_mask(self, model, dataloader, device, num_rounds):
        """
        Progressive multi-round calibration, following TaLoS ([15], Sec. 4.2).

        Round r (of R) keeps (1 - sparsity_level) ** ((r + 1) / R) of the scored weights
        trainable, so the trainable set shrinks gradually until it reaches 1 - sparsity_level.
        Each round:
        1. Shrinks excluded weights to 0.1x and recomputes the Fisher scores on that model
        2. Picks, among the weights still trainable, the lowest (least_sensitive) or
           highest (most_sensitive) scores; excluded weights never come back (global_copy)
        3. Restores the original weights

        With num_rounds=1 this is a single one-shot scoring at the final sparsity.
        """
        params = self._scored_params(model)
        keep = 1 - self.sparsity_level
        train_lowest = self.strategy == "least_sensitive"
        mask = None

        for r in range(num_rounds):
            target = keep ** ((r + 1) / num_rounds)
            scores = self._fisher_scores(model, params, dataloader, device, mask,
                                         desc=f"Fisher Round {r+1}/{num_rounds}")
            mask = self._build_mask(scores, train_lowest, keep_fraction=target, eligible=mask)
            # _build_mask reuses self.global_mask, so keep an independent copy as next eligible
            mask = {name: m.clone() for name, m in mask.items()}
            self.round_fractions.append(self.trained_fraction)

        self.global_mask = mask
        return mask

    def _fisher_scores(self, model, params, dataloader, device, mask, desc):
        """
        Fisher Information diagonal (squared gradients), averaged over samples.
        Weights excluded by mask are shrunk to 0.1x during scoring and restored afterwards.
        """
        model.eval()
        criterion = nn.CrossEntropyLoss()
        originals = {}
        if mask is not None:
            with torch.no_grad():
                for name, p in params.items():
                    originals[name] = p.detach().clone()
                    scale = torch.where(mask[name].to(p.device) == 1, 1.0, EXCLUDED_WEIGHT_SCALE)
                    p.mul_(scale)

        fisher = {name: torch.zeros_like(p) for name, p in params.items()}
        total_samples = 0
        try:
            with torch.enable_grad():
                for inputs, targets in tqdm(dataloader, desc=desc):
                    inputs, targets = inputs.to(device), targets.to(device)
                    total_samples += inputs.size(0)
                    model.zero_grad()
                    loss = criterion(model(inputs), targets)
                    loss.backward()

                    with torch.no_grad():
                        for name, p in params.items():
                            if p.grad is not None:
                                # Fisher Information: F_ii = (∂L/∂θ_i)², weighted by batch size
                                fisher[name] += p.grad.pow(2) * inputs.size(0)
        finally:
            model.zero_grad()
            with torch.no_grad():
                for name, original in originals.items():
                    params[name].copy_(original)

        return {name: f / total_samples for name, f in fisher.items()}


    def _random_mask(self, model):
        # Uniform random scores from a seeded generator, so masks are reproducible
        generator = torch.Generator().manual_seed(self.seed)
        scores = {
            name: torch.rand(p.shape, generator=generator)
            for name, p in self._scored_params(model).items()
        }
        self._build_mask(scores, train_lowest=True)
        self.round_fractions.append(self.trained_fraction)
        return self.global_mask

    def _magnitude_mask(self, model):
        scores = {
            name: p.data.abs()
            for name, p in self._scored_params(model).items()
        }
        self._build_mask(scores, train_lowest=(self.strategy == "lowest_magnitude"))
        self.round_fractions.append(self.trained_fraction)
        return self.global_mask
