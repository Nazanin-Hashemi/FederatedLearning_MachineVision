

import torch
from torch.optim import SGD
from typing import Iterable, Optional, Dict, Any, List

class SparseSGDM(SGD):
    """
    SGD with Momentum that supports sparse updates via masks.
    Mask can be:
      - None
      - dict mapping param or id(param) -> mask tensor
      - list of masks in the same order as the params
    """

    def __init__(self, params: Iterable, lr: float = 1e-3, momentum: float = 0.0,
                 dampening: float = 0, weight_decay: float = 0, nesterov: bool = False,
                 mask: Optional[Any] = None, *, maximize: bool = False):

        super().__init__(params, lr=lr, momentum=momentum,
                         dampening=dampening, weight_decay=weight_decay,
                         nesterov=nesterov, maximize=maximize)

        flat_params: List[torch.nn.Parameter] = [] # gather all params
        # Iterate over all parameter groups and parameters to flatten them into a single list
        for group in self.param_groups:
            for p in group['params']:
                flat_params.append(p)

        # Build internal map from id(p) -> mask
        self._mask_by_id: Dict[int, torch.Tensor] = {}

        if mask is None:
            pass
        
        # Handle dict case
        elif isinstance(mask, dict):
            # Iterate over items and map by id
            for k, v in mask.items():
                if isinstance(k, torch.nn.Parameter):
                    self._mask_by_id[id(k)] = v  # map by id
                else:
                    self._mask_by_id[int(k)] = v # map by id directly

        # Handle list/tuple case
        elif isinstance(mask, (list, tuple)):
            if len(mask) != len(flat_params):
                raise ValueError("Mask list length must match number of parameters.")
            for p, m in zip(flat_params, mask):
                self._mask_by_id[id(p)] = m # map by id

        else:
            raise TypeError("mask must be None, dict, or list/tuple of tensors")

    # Get mask for a parameter
    def _get_mask(self, p: torch.nn.Parameter):
        return self._mask_by_id.get(id(p), None)

    # Override step method to apply masks
    def step(self, closure=None):

        # Save original gradients
        orig_grads: List[Optional[torch.Tensor]] = []

        # Apply mask to gradients BEFORE update
        for group in self.param_groups:
            # Iterate over parameters
            for p in group['params']:
                if p.grad is None:
                    orig_grads.append(None)
                    continue
                # Save original gradient
                orig_grads.append(p.grad.clone())

                # Apply mask if exists
                mask = self._get_mask(p)
                if mask is not None:
                    # enforce boolean mask to avoid numeric drift
                    mask_bool = mask.to(p.grad.device).bool()
                    p.grad = p.grad * mask_bool

        # Let PyTorch SGD do momentum + update
        loss = super().step(closure)

        # After step, zero out masked positions in momentum buffer
        flat_index = 0
        for group in self.param_groups:
            for p in group['params']:
                mask = self._get_mask(p)
                state = self.state.get(p, {})

                # Fix momentum buffer
                if mask is not None and 'momentum_buffer' in state:
                    buf = state['momentum_buffer']
                    mask_bool = mask.to(buf.device).bool()
                    buf.mul_(mask_bool)

                # Restore original gradients
                orig = orig_grads[flat_index]
                p.grad = orig
                flat_index += 1

        return loss
    
