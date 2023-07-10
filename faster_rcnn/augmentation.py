import imgaug as ia
from imgaug import augmenters as iaa
import torch
from typing import *

class Normalize:

    def __init__(self):
        super().__init__()

    def __call__(self, input: torch.Tensor) -> torch.Tensor:

        x = torch.clone(input)

        max_value = torch.amax(x, axis=(2, 3), keepdims=True)
        min_value = torch.amin(x, axis=(2, 3), keepdims=True)
        x = (x - min_value) / (max_value - min_value)

        return x
    
class CleverCropAndResize:

    def __init__(self, n_manual = None) -> None:
        
        super().__init__()

        self.n_manual = n_manual
        

    def crop_around_bboxes(self, image: torch.Tensor, targets: Dict[str, torch.Tensor], n_manual: int = None):

        height, width = image.shape[1:]

        n_targets = len(targets["boxes"]) 

        # Select random number of consecutive vertebrae
        n = torch.randint(1, n_targets + 1, (1,)).item() if n_manual is None else n_manual

        # Select start index
        start = torch.randint(0, n_targets - n + 1, (1,)).item()

        # Select end index
        end = start + n

        # Select the targets
        bboxes = targets["boxes"][start:end]

        image_height, image_width = image.shape[1:]

        # Get the min and max x and y coordinates
        x_min = bboxes[:, 0].min().int().item()
        x_max = bboxes[:, 2].max().int().item()
        y_min = bboxes[:, 1].min().int().item()
        y_max = bboxes[:, 3].max().int().item()

        # Random expansion within the image
        max_expansion = 0.2

        # Keeping aspect ratio, expand the bounding box
        expansion = torch.rand(1).item() * max_expansion
        x_min = max(0, x_min - int(expansion * (x_max - x_min)))
        x_max = min(image_width, x_max + int(expansion * (x_max - x_min)))
        y_min = max(0, y_min - int(expansion * (y_max - y_min)))
        y_max = min(image_height, y_max + int(expansion * (y_max - y_min)))

        # Crop the image
        cropped_image = image[:, y_min:y_max, x_min:x_max]

        # Resize the image to original size
        old_height, old_width = cropped_image.shape[1:]
        cropped_image = torch.nn.functional.interpolate(cropped_image.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False).squeeze(0)

        # Crop the targets and resize them
        cropped_targets = {
            "boxes": targets["boxes"][start:end] - torch.tensor([x_min, y_min, x_min, y_min], device=targets["boxes"].device),
            "keypoints": targets["keypoints"][start:end] - torch.tensor([x_min, y_min, 0], device=targets["keypoints"].device),
            "labels": targets["labels"][start:end],
        }

        cropped_targets["boxes"][:, 0] *= width / old_width
        cropped_targets["boxes"][:, 1] *= height / old_height
        cropped_targets["boxes"][:, 2] *= width / old_width
        cropped_targets["boxes"][:, 3] *= height / old_height

        cropped_targets["keypoints"][:, :, 0] *= width / old_width
        cropped_targets["keypoints"][:, :, 1] *= height / old_height
  
        return cropped_image, cropped_targets

    def __call__(self, images: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        
        cropped_images = []
        cropped_targets = []
        for image, target in zip(images, targets):
            cropped_image, cropped_target = self.crop_around_bboxes(image, target, n_manual=self.n_manual)
            cropped_images.append(cropped_image)
            cropped_targets.append(cropped_target)

        return torch.stack(cropped_images), cropped_targets
    

class DetectionAugmentation:

    def __init__(self, 
                 p: float = 0.5,
                 crop_percent: Tuple[float, float] = (0, 0.3), 
                 p_dropout: float = 0.2,
                 p_resize: float = 0.0,
                 p_crop: float = 0.5,
                 size_percent: Tuple[float, float] = (0.001, 0.002), 
                 blur_sigma: Tuple[float, float] = (1.0, 1.0),
                 clip_limit: Tuple[float, float] = (0.1, 8.0),
                 tile_grid_size: Tuple[int, int] = (3, 12),
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.normalize = Normalize()
        self.clever_crop = CleverCropAndResize()
        self.p = p
        self.p_resize = p_resize
        self.p_crop = p_crop
        self.p_dropout = p_dropout
        self.crop_percent = crop_percent
        self.size_percent = size_percent
        self.blur_sigma = blur_sigma
        
        self.augment = iaa.Sequential([
                iaa.Sometimes(p=p_resize, then_list=[
                iaa.KeepSizeByResize(
                    iaa.Crop(percent=crop_percent, keep_size=False),
                )]),
                # iaa.CLAHE(clip_limit=clip_limit, tile_grid_size_px=tile_grid_size),
                iaa.GaussianBlur(blur_sigma),
                # iaa.SaltAndPepper(p=p_dropout),
                # iaa.Invert(p=p),
                iaa.Sometimes(p=p, then_list=[
                iaa.Dropout(p=p_dropout)
                ]),
                iaa.Fliplr(p=p),
                iaa.Flipud(p=p),
                iaa.Sometimes(p=p, then_list=[
                iaa.CoarseDropout(p=p_dropout, size_percent=size_percent)
                ]),
            ])
        
    def __call__(self, x: torch.Tensor, targets: Dict[str, torch.Tensor]) -> torch.Tensor:


        device = x.device

        box_set = []
        keypoint_set = []
        
        p = torch.rand(1).item()

        if p < self.p_crop:

            x, targets = self.clever_crop(x, targets)


        for y in targets:
            bboxes      = y["boxes"].cpu().numpy()
            annotations   = y["keypoints"].cpu().numpy()
            n_keypoints = annotations.shape[1]
            
            kps = []
            bbs = []
            for bb in bboxes:
                if len(bb) > 0:
                    bbs.append(ia.BoundingBox(x1=bb[0], y1=bb[1], x2=bb[2], y2=bb[3]))

            # Add directly to keypoint set
            for ann in annotations:
                for kp in ann:
                    if len(kp) > 0:
                        kps.append(ia.Keypoint(x=kp[0], y=kp[1]))
            
            bbs_oi = ia.BoundingBoxesOnImage(bbs, shape=x.shape[2:])
            kps_oi = ia.KeypointsOnImage(kps, shape=x.shape[2:])

            box_set.append(bbs_oi)
            keypoint_set.append(kps_oi)


        # Reshape from (B, C, H, W) to (B, H, W, C)
        x = x.permute(0, 2, 3, 1).cpu().numpy()
        
        x_aug, bbs_aug, kps_aug = self.augment(images=x, bounding_boxes=box_set, keypoints=keypoint_set)
        # Reshape from (B, H, W, C) to (B, C, H, W)
        x_aug = torch.from_numpy(x_aug).permute(0, 3, 1, 2).to(device)

        # Normalize to float32
        x_aug = self.normalize(x_aug)

        boxes = [torch.tensor([[bb.x1, bb.y1, bb.x2, bb.y2] for bb in bb_aug], dtype=torch.float32).to(device) for bb_aug in bbs_aug]

        keypoints = []
        for kp_aug in kps_aug:
            vertebrae = [kp_aug[i:i+n_keypoints] for i in range(0, len(kp_aug), n_keypoints)]
            vs = []
            for vertebra in vertebrae:
                kps = []
                for kp in vertebra:
                    kps.append([
                        kp.x,
                        kp.y,
                        1
                    ])
                vs.append(kps)
            keypoints.append(vs)

        
        # keypoints = torch.tensor(keypoints, dtype=torch.float32).to(device)

        targets_aug = [
            {
                "boxes": box,
                "keypoints": torch.tensor(keypoint_group, dtype=torch.float32).to(device),
                "labels": y["labels"],
            }
            for box, keypoint_group, y in zip(boxes, keypoints, targets)
        ]
    
        return x_aug, targets_aug