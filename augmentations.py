import torch
import torch.nn as nn
from torchvision import transforms as T
from kornia import augmentation as K
from typing import *
import preprocessing.images as imaging


class Patchify(nn.Module):

    def __init__(self, resize_shape: Tuple[int, int] , patch_size: int = 256):
        super().__init__()
        self.resize_shape = resize_shape
        self.patch_size = patch_size
        self.resize = T.Resize(self.resize_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.resize(x)

        return self.patchify(x, self.patch_size)
    
    def patchify(self, image: torch.Tensor, patch_size: int) -> torch.Tensor:

        image   = image.unfold(1, *patch_size).unfold(2, *patch_size).reshape(-1, *patch_size)

        return image

class Normalize(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x)
    
    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        
        return imaging.normalize(image)

class Augmenter(nn.Module):

    def __init__(self, p: float = 0.5):
        super().__init__()

        self.p = p

        self.augment = K.AugmentationSequential(
            Normalize(),
            K.RandomHorizontalFlip(p=p),
            K.RandomEqualize(p=p),
            K.RandomSharpness(sharpness=(0.5, 2.0), p=p),
            K.RandomContrast(contrast=(0.5, 2.0), p=p),
            K.RandomErasing(p=p),
            same_on_batch=False
        )

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        
        return imaging.normalize(image)

    @torch.no_grad()
    def forward(self, x):
        return self.augment(x)