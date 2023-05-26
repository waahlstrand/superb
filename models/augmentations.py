import torch
import torch.nn as nn
from torchvision import transforms as T
import torchvision.transforms.functional as F
from kornia import augmentation as K
import kornia
from typing import *
import numpy as np
from PIL import Image

def threshold(img: np.ndarray, window_width: float, window_level: float) -> np.ndarray:
    """Thresholds an image based on a window width and window center.
    Args:
        image (np.ndarray): The image to threshold.
        window_width (float): The window width.
        window_center (float): The window center.
    Returns:
        np.ndarray: The thresholded image.
    """
    image = img.copy()
    image = image.squeeze()
    # image = image - window_level
    max_value = window_level + window_width / 2
    min_value = window_level - window_width / 2
    
    image[image < min_value] = min_value
    image[image > max_value] = max_value

    return image


class Preprocess(nn.Module):

    def __init__(self, target_size: Tuple[int, int], dtype: np.dtype = np.float32):
        super().__init__()
        self.target_size = target_size
        self.dtype = dtype
        
        self.processing = T.Compose([
            Normalize(dtype=self.dtype),
            ToTensor(),
            T.Lambda(lambda x: x.repeat(3, 1, 1) )
        ])

    def forward(self, x: str) -> torch.Tensor:

        x = Image.open(x).resize(self.target_size[::-1]) 
        x = np.array(x)

        x = self.processing(x)

        return x

class Contrast(nn.Module):

    def __init__(self, w: Tuple[int, int], l: Tuple[int, int]):
        super().__init__()
        self.w_range = w
        self.l_range = l 

    def forward(self, x: np.ndarray) -> np.ndarray:

        w = np.random.randint(*self.w_range, size=x.shape[0])
        l = np.random.randint(*self.l_range, size=x.shape[0])

        for i in range(x.shape[0]):
            x[i] = threshold(x[i], w[i], l[i])

        return x

class Patchify(nn.Module):

    def __init__(self, patch_size: Tuple[int, int]):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.patchify(x, self.patch_size)
    
    def patchify(self, image: torch.Tensor, patch_size: int) -> torch.Tensor:

        batch_size, channels, height, width = image.shape
        image   = image.unfold(2, *patch_size).unfold(3, *patch_size).reshape(batch_size, -1, *patch_size)

        return image

class Normalize(nn.Module):

    def __init__(self, dtype: np.dtype = np.float32):
        super().__init__()

        self.dtype = dtype

    def forward(self, input: np.ndarray) -> np.ndarray:

        x = input.copy()
        
        x = x.astype(np.float64)

        if self.dtype == np.float32:
            max_value = np.amax(x, axis=(0, 1), keepdims=True)
            min_value = np.amin(x, axis=(0, 1), keepdims=True)
            x = (x - min_value) / (max_value - min_value)
        elif self.dtype == np.uint8:
            max_value = np.amax(x, axis=(0, 1), keepdims=True)
            min_value = np.amin(x, axis=(0, 1), keepdims=True)
            x = (x - min_value) / (max_value - min_value)
            x = (x * 255).astype(np.uint8)

        x = x.astype(self.dtype)

        return x
    
class ToTensor(nn.Module):
    
        def __init__(self):
            super().__init__()
            
    
        def forward(self, x: np.ndarray) -> torch.Tensor:

            x = torch.from_numpy(x)
    
            return x
    
class CLAHE(nn.Module):
    
        def __init__(self, clipLimit: float = 0.9, tileGridSize: Tuple[int, int] = (8, 8)):
            super().__init__()
            self.clipLimit = clipLimit
            self.tileGridSize = tileGridSize
    
        def forward(self, x: torch.Tensor) -> torch.Tensor:
    
            x = kornia.enhance.equalize_clahe(x, self.clipLimit, self.tileGridSize)

            return x
        

class SameRandomCrop(nn.Module):

    def __init__(self, size: Tuple[int, int] = (256, 256), p: float = 0.5, expansion_factor: float = 0.3):
        super().__init__()
        self.size = size
        self.p = p
        self.resize = T.Resize(size)
        self.expansion_factor = expansion_factor

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size = x1.shape[0]

        x1_crop = torch.zeros((batch_size, x1.shape[1], self.size[0], self.size[1])).to(x1.device)
        x2_crop = torch.zeros((batch_size, x1.shape[1], self.size[0], self.size[1])).to(x2.device)

        for idx in range(batch_size):

            if np.random.rand() > self.p:
                x1_crop[idx,:,:,:] = self.resize(x1[idx,:,:,:])
                x2_crop[idx,:,:,:] = self.resize(x2[idx,:,:,:])

                continue

            i, j, h, w = T.RandomCrop.get_params(x1[idx, :, :, :], output_size=self.size)

            # Randomly expand the crop
            offset = int(self.expansion_factor * np.abs(h - i)) 
            pixel_expansion = offset

            # Crop the first image
            x1_crop[idx, :, :, :] = F.crop(x1[idx, :, :, :], i, j, h, w)

            # print(i, j, h, w, pixel_expansion)
            # Randomly expand the crop 
            i = max(0, i - pixel_expansion)
            j = max(0, j - pixel_expansion)
            h = min(x2.shape[2], h + pixel_expansion)
            w = min(x2.shape[3], w + pixel_expansion)



            # print(i, j, h, w)

            # Crop the second image            
            x2_crop[idx, :, :, :] = self.resize(F.crop(x2[idx, :, :, :], i, j, h, w))


        return x1_crop, x2_crop


class DetectionAugmentation(nn.Module):

    def __init__(self, p: float = 0.5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.p = p
        
        self.augment = K.AugmentationSequential(
                # K.RandomCrop(size=self.crop_size, p=p),
                T.RandomApply([CLAHE()], p=p),
                K.RandomGaussianBlur(kernel_size=(11,11), sigma=(1.0, 1.0), p=p),
                K.RandomInvert(p=p),
                # K.RandomHorizontalFlip(p=p),
                # K.RandomVerticalFlip(p=p),
                # K.Resize(size=initial_size),
                same_on_batch=False
            )
        
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
            
        x = self.augment(x)
    
        return x       

class Augmentation(nn.Module):

    def __init__(self, initial_size=Tuple[int,int], crop_size: Tuple[int, int] = (256, 256), p: float = 0.5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.p = p
        self.crop_size = crop_size
        
        self.augment = K.AugmentationSequential(
                K.RandomCrop(size=self.crop_size, p=p),
                T.RandomApply([CLAHE()], p=p),
                K.RandomGaussianBlur(kernel_size=(11,11), sigma=(1.0, 1.0), p=p),
                K.RandomInvert(p=p),
                K.RandomHorizontalFlip(p=p),
                K.RandomVerticalFlip(p=p),
                K.Resize(size=initial_size),
                same_on_batch=False
            )
        
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
            
        x = self.augment(x)
    
        return x
    

class SimSiamAugmentation(nn.Module):

    def __init__(self, p: float = 0.5, n: int = 2, m: int = 10, size=(256, 256), *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.p = p
        self.n = n
        self.m = m
        self.size = size

        self.augment = K.AugmentationSequential(
                K.RandomCrop(size=size, p=p),
                K.auto.RandAugment(n=n, m=m),
                same_on_batch=False
            )
        
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
            
        x = self.augment(x)

        return x
        
class Augmenter(nn.Module):

    def __init__(self, p: float = 0.5, mode: str = 'train'):
        super().__init__()

        self.p = p
        self.mode = mode

        self.processing = T.Compose([
            Normalize(),
            ToTensor(),
            T.Lambda(lambda x: x.repeat(3, 1, 1) )
        ])

        self.patchify = Patchify(patch_size=(256, 256))


        self.contrast = T.RandomApply(
                    [Contrast(w=(1500, 3850), l=(1726, 2500))],
                    p=p
            )

        self.augment = K.AugmentationSequential(
                T.RandomApply([CLAHE()], p=p),
                K.RandomHorizontalFlip(p=p),
                K.RandomVerticalFlip(p=p),
                same_on_batch=False
            )


    @torch.no_grad()
    def forward(self, x: np.ndarray) -> torch.Tensor:

        t = torch.zeros(x.shape[0], 3, x.shape[2], x.shape[3])

        if self.mode == 'train':
            for i in range(x.shape[0]):
                c = self.contrast(x[i, :, :, :])
                # c = x[i, :, :, :]
                t[i, :, :, :] = self.processing(c)

            # t = self.augment(t)
            # t = self.patchify(t)

            return t

        elif self.mode == 'val':
            
            for i in range(t.shape[0]):
                t[i, :, :, :] = self.processing(x[i, :, :, :])

            # t = self.patchify(t)

            return t
        else:
            raise ValueError(f'Invalid mode: {self.mode}')
