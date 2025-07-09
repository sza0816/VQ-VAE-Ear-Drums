import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile


# Add your custom dataset class here
class EarDrumDataset(Dataset):
    def __init__(
        self, 
        root, 
        split = 'train', 
        transform = None):

        self.data_dir = Path(root)/'Normal'             # use class Normal of root
        self.transforms = transform

        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix in ['.png']])
        
        if split == 'train':
            self.imgs = imgs[:int(len(imgs)*0.75)]
        else: 
            self.imgs = imgs[int(len(imgs)*0.75):]
        print(f"[Dataset] check dataset size: {len(self.imgs)}")                       # check number of images
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        if self.transforms:
            img = self.transforms(img)
        return img, 0.0

class GaussianNoise(object):  
    def __init__(self, mean=0.0, std=1.0):  
        self.mean = mean 
        self.std = std 
    def __call__(self, tensor): 
    	noise = torch.randn(tensor.size()) * self.std + self.mean 
    	return tensor + noise 
 
class VAEDataset(LightningDataModule):                        # imported to run.py
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        # train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                       transforms.CenterCrop(148),
        #                                       transforms.Resize(self.patch_size),
        #                                       transforms.ToTensor(),])                   # can add normalize later
        
        # val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                     transforms.CenterCrop(148),
        #                                     transforms.Resize(self.patch_size),
        #                                     transforms.ToTensor(),])
        train_transforms = transforms.Compose([ 
                                        transforms.RandomResizedCrop(64, scale=(0.8, 1.0), ratio=(0.9, 1.1)), 
                                        transforms.RandomRotation(degrees=(-10, 10)), 
                                        transforms.ColorJitter(brightness=0.2, contrast=0.2), 
                                        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5), 
                                        transforms.ToTensor(), 
                                        transforms.RandomApply([GaussianNoise(mean=0, std=0.05)], p=0.5), 
        ]) 

        val_transforms = transforms.Compose([ 
                                        transforms.Resize((64, 64)), 
                                        transforms.ToTensor(), 
                                        ]) 

        # replace MyCelebA dataset with EarDrumDataset
        self.train_dataset = EarDrumDataset(root = self.data_dir, split='train', transform=train_transforms)
        self.val_dataset = EarDrumDataset(root = self.data_dir, split='test', transform=val_transforms)
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
     