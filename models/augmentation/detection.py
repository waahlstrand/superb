import imgaug as ia
from imgaug import augmenters as iaa
import torch
from typing import *
import numpy as np

class Normalize:

    def __init__(self):
        super().__init__()

    def __call__(self, input: torch.Tensor) -> torch.Tensor:

        x = torch.clone(input)

        max_value = torch.amax(x, axis=(2, 3), keepdims=True)
        min_value = torch.amin(x, axis=(2, 3), keepdims=True)
        x = (x - min_value) / (max_value - min_value)

        return x

class DetectionAugmentation:

    def __init__(self, 
                 p: float = 0.5,
                 crop_percent: Tuple[float, float] = (0, 0.3), 
                 p_dropout: float = 0.2,
                 size_percent: Tuple[float, float] = (0.001, 0.002), 
                 blur_sigma: Tuple[float, float] = (1.0, 1.0),
                 clip_limit: Tuple[float, float] = (0.1, 8.0),
                 tile_grid_size: Tuple[int, int] = (3, 12),
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.normalize = Normalize()

        self.p = p
        self.p_dropout = p_dropout
        self.crop_percent = crop_percent
        self.size_percent = size_percent
        self.blur_sigma = blur_sigma
        
        self.augment = iaa.Sequential([
                iaa.KeepSizeByResize(
                    iaa.Crop(percent=crop_percent, keep_size=False),
                ),
                # iaa.CLAHE(clip_limit=clip_limit, tile_grid_size_px=tile_grid_size),
                iaa.GaussianBlur(blur_sigma),
                iaa.Invert(p=p),
                iaa.Fliplr(p=p),
                iaa.Flipud(p=p),
                iaa.Sometimes(p=p, then_list=[
                iaa.CoarseDropout(p=p_dropout, size_percent=size_percent)
                ]),
            ])
        
    def __call__(self, x: torch.Tensor, targets: Dict[str, torch.Tensor]) -> torch.Tensor:


        device = x.device

        boxes = []

        for y in targets:
            bboxes = y["boxes"].cpu().numpy()
            bbs = []
            for bb in bboxes:
                if len(bb) > 0:
                    bbs.append(ia.BoundingBox(x1=bb[0], y1=bb[1], x2=bb[2], y2=bb[3]))
            boxes.append(bbs)
            
    

        # Reshape from (B, C, H, W) to (B, H, W, C)
        x = x.permute(0, 2, 3, 1).cpu().numpy()
            
        x_aug, bbs_aug = self.augment(images=x, bounding_boxes=boxes)

        # Reshape from (B, H, W, C) to (B, C, H, W)
        x_aug = torch.from_numpy(x_aug).permute(0, 3, 1, 2).to(device)

        # Normalize to float32
        x_aug = self.normalize(x_aug)


        targets_aug = [
            {
                "boxes": torch.tensor([[bb.x1, bb.y1, bb.x2, bb.y2] for bb in bbs_aug], dtype=torch.float32).to(device) if len(bbs_aug) > 0 else torch.empty((0, 4), dtype=torch.float32).to(device),
                "labels": y["labels"],
            }
            for bbs_aug, y in zip(bbs_aug, targets)
        ]
    
        return x_aug, targets_aug