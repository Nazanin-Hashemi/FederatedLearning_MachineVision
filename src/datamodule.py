import os
from typing import Optional, Literal, Dict, List

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class TransformSubset(Dataset):    
    def __init__(self, dataset: Dataset, indices: list, transform: Optional[transforms.Compose] = None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    
    def __getitem__(self, idx: int):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.indices)



class BaseCIFAR100DataModule(pl.LightningDataModule):    
    MEAN = (0.5071, 0.4867, 0.4408)
    STD = (0.2675, 0.2565, 0.2761)
    
    def __init__(
        self,
        data_dir: str = './data',
        batch_size: int = 128,
        num_workers: Optional[int] = None,
        val_split: float = 0.1,
        seed: int = 42,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else min(4, os.cpu_count() or 0)
        self.val_split = val_split
        self.seed = seed
        self.image_size = image_size
    
    @property
    def train_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])
    
    @property
    def test_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])
    
    def prepare_data(self) -> None:
        datasets.CIFAR100(self.data_dir, train=True, download=True)
        datasets.CIFAR100(self.data_dir, train=False, download=True)
    
    def setup(self, stage: Optional[str] = None) -> None:
        """This is the shared logic for both child classes, note the super() call in these"""
        if stage == 'fit' or stage is None:
            self.full_train = datasets.CIFAR100(self.data_dir, train=True, download=False)
            n_train = int(len(self.full_train) * (1 - self.val_split))
            
            generator = torch.Generator().manual_seed(self.seed)
            indices = torch.randperm(len(self.full_train), generator=generator).tolist()
            self.train_indices = indices[:n_train]
            
            self.val_dataset = TransformSubset(self.full_train, indices[n_train:], self.test_transform)
        
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.CIFAR100(
                self.data_dir, train=False, download=False, transform=self.test_transform
            )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )






class CIFAR100DataModule(BaseCIFAR100DataModule):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()
        
    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage)
        if stage == 'fit' or stage is None:
            self.train_dataset = TransformSubset(
                self.full_train, self.train_indices, self.train_transform
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0, drop_last=True,
        )
    





class FederatedCIFAR100DataModule(BaseCIFAR100DataModule):    
    def __init__(
        self,
        num_clients: int = 100,
        sharding: Literal['iid', 'non_iid'] = 'iid',
        num_classes_per_client: int = 10,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.num_clients = num_clients
        self.sharding = sharding
        self.num_classes_per_client = num_classes_per_client
        self.client_indices: Dict[int, List[int]] = {}
        self.save_hyperparameters()
    
    def _iid_sharding(self, targets: np.ndarray) -> Dict[int, List[int]]:
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(len(targets))
        splits = np.array_split(indices, self.num_clients)
        return {i: split.tolist() for i, split in enumerate(splits)}
    
    def _non_iid_sharding(self, targets: np.ndarray) -> Dict[int, List[int]]:
        rng = np.random.default_rng(self.seed)
        num_classes = len(np.unique(targets))
        
        class_indices = {c: np.where(targets == c)[0].tolist() for c in range(num_classes)}
        for c in class_indices:
            rng.shuffle(class_indices[c])
        
        total_shards = self.num_clients * self.num_classes_per_client
        shards_per_class = total_shards // num_classes
        
        all_shards = []
        for c in range(num_classes):
            class_data = class_indices[c]
            shard_size = len(class_data) // shards_per_class
            for s in range(shards_per_class):
                start = s * shard_size
                end = start + shard_size if s < shards_per_class - 1 else len(class_data)
                all_shards.append(class_data[start:end])
        
        rng.shuffle(all_shards)
        
        client_indices = {i: [] for i in range(self.num_clients)}
        for client_id in range(self.num_clients):
            for j in range(self.num_classes_per_client):
                shard_idx = client_id * self.num_classes_per_client + j
                if shard_idx < len(all_shards):
                    client_indices[client_id].extend(all_shards[shard_idx])
            rng.shuffle(client_indices[client_id])
        
        return client_indices
    
    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage)
        
        if stage == 'fit' or stage is None:
            targets = np.array([self.full_train.targets[i] for i in self.train_indices])
            
            if self.sharding == 'iid':
                relative_indices = self._iid_sharding(targets)
            else:
                relative_indices = self._non_iid_sharding(targets)
            
            self.client_indices = {
                cid: [self.train_indices[i] for i in idx_list]
                for cid, idx_list in relative_indices.items()
            }
    
    def get_client_dataloader(self, client_id: int, shuffle: bool = True) -> DataLoader:
        client_dataset = TransformSubset(
            self.full_train, self.client_indices[client_id], self.train_transform
        )
        return DataLoader(
            client_dataset, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=True,
            drop_last=len(client_dataset) > self.batch_size,
        )
    
    def get_client_sample_count(self, client_id: int) -> int:
        return len(self.client_indices[client_id])
    
    # might not be necessary
    def get_client_class_distribution(self, client_id: int) -> Dict[int, int]:
        indices = self.client_indices[client_id]
        targets = [self.full_train.targets[i] for i in indices]
        unique, counts = np.unique(targets, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    

    def train_dataloader(self) -> DataLoader:
        """just used for ridge, might change to use per client data"""
        all_indices = [idx for indices in self.client_indices.values() for idx in indices]
        full_dataset = TransformSubset(self.full_train, all_indices, self.test_transform)
        return DataLoader(
            full_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )
