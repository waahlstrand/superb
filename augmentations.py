import torch
import torch.nn as nn
from torchvision import transforms as T
from kornia import augmentation as K
import kornia
from typing import *
import preprocessing.images as imaging
import cv2

class Preprocess(nn.Module):

    def __init__(self, target_size: Tuple[int, int] = imaging.PADDING_SHAPE):
        super().__init__()
        self.target_size = target_size
        self.toTensor = T.ToTensor()
        self.normalize = lambda x: cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.read_from_path = lambda x: cv2.imread(x, -1)
        self.blur = lambda x: cv2.GaussianBlur(x, (5, 5), 0)
        self.clahe = cv2.createCLAHE(clipLimit = 4, tileGridSize = (8, 8))
        self.resize = lambda x: cv2.resize(x, self.target_size[::-1], interpolation = cv2.INTER_AREA)

    def forward(self, x: str) -> torch.Tensor:

        x = self.read_from_path(x)
        x = self.normalize(x)
        # x = self.clahe.apply(x)
        # x = self.blur(x)
        x = self.resize(x)
        x = self.toTensor(x)
        x = x.squeeze()

        return x

class Patchify(nn.Module):

    def __init__(self, resize_shape: Tuple[int, int] , patch_size: Tuple[int, int]):
        super().__init__()
        self.resize_shape = resize_shape
        self.patch_size = patch_size
        self.resize = T.Resize(self.resize_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.patchify(x, self.patch_size)
    
    def patchify(self, image: torch.Tensor, patch_size: int) -> torch.Tensor:

        batch_size, channels, height, width = image.shape
        image   = image.unfold(2, *patch_size).unfold(3, *patch_size).reshape(batch_size, -1, *patch_size)

        return image

class Normalize(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x)
    
    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        
        return imaging.normalize(image)
    
class CLAHE(nn.Module):
    
        def __init__(self, clipLimit: float = 0.8, tileGridSize: Tuple[int, int] = (8, 8)):
            super().__init__()
            self.clipLimit = clipLimit
            self.tileGridSize = tileGridSize
    
        def forward(self, x: torch.Tensor) -> torch.Tensor:
    
            x = kornia.enhance.equalize_clahe(x, self.clipLimit, self.tileGridSize)

            return x
        
class Augmenter(nn.Module):

    def __init__(self, p: float = 0.5):
        super().__init__()

        self.p = p

        self.augment = K.AugmentationSequential(
            # T.RandomApply([CLAHE()], p=p),
            K.RandomHorizontalFlip(p=p),
            K.RandomVerticalFlip(p=p),
            K.RandomCrop((512, 512), p=1),
            T.RandomApply([CLAHE()], p=p),
            # K.RandomRotation(degrees=15, p=p),
            # K.RandomEqualize(p=p),
            # K.RandomSharpness(sharpness=(0.5, 2.0), p=p),
            # K.RandomContrast(contrast=(0.5, 2.0), p=p),
            # K.RandomErasing(p=p),
            # Patchify((256, 256), (256, 256)),
            # K.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), p=p),
            same_on_batch=False
        )

    @torch.no_grad()
    def forward(self, x):

        # x = super().forward(x)

        return self.augment(x)