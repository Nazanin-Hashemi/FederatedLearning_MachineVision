"""
Sparse Stochastic Gradient Descent with Momentum (SparseSGDM) Optimizer

Implements Step 2 of sparse fine-tuning: performs gradient descent with per-parameter
gradient masking. Masked parameters (mask=0) are frozen at their pre-trained values
by zeroing their gradients and momentum buffers.

This extends the standard SGDM optimizer to support element-wise gradient masks,
allowing fine-grained control over which parameter elements are updated during training.

Reference: Task Arithmetic methodology for sparse fine-tuning in federated learning.
"""
import torch
from torch.optim import SGD

class SparseSGDM(SGD):
    """
    SGD with Momentum that supports element-wise gradient masks for sparse fine-tuning.
    
    During optimization:
    - Parameters with mask=1: Receive gradient updates (sparse fine-tuning)
    - Parameters with mask=0: Gradients masked to zero (frozen at pre-trained values)
    
    The optimizer ensures masked parameters preserve their pre-trained values by:
    1. Masking gradients before optimizer step
    2. Masking weight decay contribution (if enabled)
    3. Masking momentum buffer after update (prevents drift)
    
    Args:
        params: Model parameters to optimize
        lr: Learning rate
        momentum: Momentum factor (default: 0.9)
        dampening: Momentum dampening factor
        weight_decay: Weight decay (L2 regularization) coefficient
        nesterov: Enable Nesterov momentum
        mask: Dictionary mapping parameter objects to binary mask tensors.
              Mask shape must match parameter shape element-wise.
              mask[p][i] = 1 means parameter element p[i] will be updated
              mask[p][i] = 0 means parameter element p[i] will be frozen
    """
    def __init__(self, params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False, mask=None):
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        # mask: dictionary mapping Parameter -> binary mask tensor (same shape as parameter)
        # mask[p] = tensor of 1s (update) and 0s (freeze) with same shape as parameter p
        self.mask = mask

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step with gradient masking.
        
        The step process:
        1. Apply gradient mask: Zero gradients at positions where mask=0
        2. Apply weight decay mask: Prevent weight decay on frozen parameters
        3. Standard SGDM update: Update weights using masked gradients
        4. Mask momentum buffer: Zero momentum at frozen positions (prevents drift)
        
        Frozen parameters (mask=0) preserve their pre-trained values throughout training.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            
            # Apply gradient masks and handle weight decay for masked parameters
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Apply element-wise gradient mask
                if self.mask is not None and p in self.mask:
                    mask_tensor = self.mask[p].to(p.grad.device)
                    
                    # Handle weight decay: Apply manually and mask it
                    # This ensures masked parameters don't shrink due to weight decay
                    if weight_decay != 0:
                        # Add weight decay term to gradient
                        p.grad.add_(p, alpha=weight_decay)
                    
                    # Zero out gradients at masked positions (mask=0 → frozen)
                    # mask=1: gradient preserved (will update)
                    # mask=0: gradient zeroed (frozen at pre-trained value)
                    p.grad.mul_(mask_tensor)

            # Temporarily disable weight_decay in base SGD optimizer
            # We've already handled it manually with masking above
            original_wd = group['weight_decay']
            group['weight_decay'] = 0
            
            # Perform standard SGDM update using masked gradients
            # This updates weights: p.data = p.data - lr * (masked_gradient)
            super().step()
            
            # Restore original weight decay setting
            group['weight_decay'] = original_wd
            
            # CRITICAL FOR TASK ARITHMETIC: 
            # Zero out the momentum buffer at masked positions to prevent "drift"
            # Masked parameters (mask=0) are frozen at their pre-trained values
            # We mask the momentum buffer to prevent accumulation of momentum at frozen positions
            for p in group['params']:
                if self.mask is not None and p in self.mask:
                    state = self.state[p]
                    if 'momentum_buffer' in state:
                        mask_tensor = self.mask[p].to(p.device)
                        state['momentum_buffer'].mul_(mask_tensor)
        
        return loss
