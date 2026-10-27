from typing import Literal

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
from sklearn.linear_model import RidgeClassifier
import numpy as np

# Model definition using DINO backbone
class DinoClassifier(pl.LightningModule):    
    def __init__(
        self, 
        num_classes: int = 100, # CIFAR-100
        lr: float = 0.01, # learning rate
        momentum: float = 0.9, # SGD momentum
        weight_decay: float = 5e-4, # weight decay
        scheduler: Literal['cosine', 'step', 'none'] = 'cosine', # lr scheduler type
        max_epochs: int = 100,  # max epochs for scheduler 
        freeze_backbone: bool = True, # whether to freeze DINO backbone
    ) -> None:
        super().__init__()
        # args are used through self.hparams, they are not unused so don't remove them
        self.save_hyperparameters()
        
        # loading pre trained dino blackbone
        self.backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        self.embed_dim = self.backbone.embed_dim # features dimension
        
        # To only train the classification head because DINO is already trained
        if freeze_backbone:
            # double check that this freezes completely
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Classifier head
        self.classifier = nn.Linear(self.embed_dim, num_classes) # final layer
        nn.init.normal_(self.classifier.weight, std=0.01) # init weights
        nn.init.zeros_(self.classifier.bias) # init bias
        
        # Loss and metrics
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)
    
    # Forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through backbone and classifier head.
        1. Extract features using the DINO backbone if frozen.
        2. Classify features using the linear classifier.
        3. Return class logits.
        """
        if self.hparams.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)
    
        return self.classifier(features)
    
    # Training step
    def training_step(self, batch, batch_idx):
        x, y = batch # unpack batch
        logits = self(x) # forward pass give logits (blackbone+classifier)
        loss = self.criterion(logits, y) # compute loss (cross entropy)
        self.train_acc(logits, y) # update accuracy metric
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # Then return loss for auto backpropagation
        # - loss.backward() : Calculate gradient for all parameters with requires_grad=True
        # - optimizer.step() : Update weights based on calculated gradients
        # - optimizer.zero_grad() : Reset gradients for next iteration
        return loss 
    
    # Validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x) 
        loss = self.criterion(logits, y) 
        self.val_acc(logits, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
    
    # Test step
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.test_acc(logits, y)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
    
    # Configure optimizers and learning rate schedulers
    def configure_optimizers(self):
        """
        Set up optimizer and learning rate scheduler.
        """
        params = filter(lambda p: p.requires_grad, self.parameters()) # only parameters to train (ie. classifier if frozen)
        
        # Classic Optimizer with SGD and our hyperparameters
        optimizer = torch.optim.SGD(
            params,
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )

        # Learning rate scheduler
        # Cosine Annealing LR scheduler : learning rate follows a cosine decay
        if self.hparams.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6
            )
        # Step LR scheduler as alternative
        # Step decay : reduce lr by gamma every step_size epochs
        elif self.hparams.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
        # No scheduler
        else:
            return optimizer
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        } # read docs, if no scheduler, only return optimizer
    

    def initialize_head_with_ridge(self, dataloader, device):
        self.eval()
        self.to(device)
        features, labels = [], []

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                # Extract features from backbone
                feat = self.backbone(inputs)
                features.append(feat.cpu().numpy())
                labels.append(targets.numpy())
        
        X = np.concatenate(features)
        y = np.concatenate(labels)

        ridge = RidgeClassifier(alpha=1.0)
        ridge.fit(X, y)

        with torch.no_grad(): # double check type conversion, unsure about that
            self.classifier.weight.copy_(torch.from_numpy(ridge.coef_.astype(np.float32)))
            self.classifier.bias.copy_(torch.from_numpy(ridge.intercept_.astype(np.float32)))

        # Freezing head
        for param in self.classifier.parameters():
            param.requires_grad = False
