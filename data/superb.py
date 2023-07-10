import pytorch_lightning as L
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.ops import box_convert, clip_boxes_to_image
from typing import *
from pathlib import Path
from utils.typings import Patient
from enum import Enum
import numpy as np
from torchvision import transforms as T
import imgaug as ia
import cv2
from dataclasses import dataclass

@dataclass
class SuperbStatistics:
    MEAN: float = 2048.724442444413
    STD: float = 375.3545219751161
    MAX_HEIGHT: int = 1656
    MAX_WIDTH: int = 603


class Compression(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3

class Stage(str, Enum):

    FIT     = 'fit'
    TRAIN   = 'train'
    VAL     = 'val'
    TEST    = 'test'

class Target(str, Enum):

    LABEL    = "labels"
    BBOX     = "boxes"
    KEYPOINT = "keypoints"

class SuperbData(Dataset):

    def __init__(self, 
                 patients_root: Path,
                 removed: List[str] = [],
                 size: Tuple[float, float] = (600, 280),
                 patient_dirs: List[Path] = [],
                 bbox_expansion: float = 0.1,
                 bbox_format: str = 'xyxy',
                 target_format: List[Target] = [Target.LABEL, Target.BBOX, Target.KEYPOINT],
                 transforms: List[Callable] = [
                 ]
                 ) -> None:
        super().__init__()

        self.patients_root = patients_root # Root directory of all patient directories
        self.removed = removed # List of patient directories to remove
        self.target_format = target_format # Format of targets to return

        # Get all patient directories, unless specified
        self.patient_dirs = [
            patient_dir for patient_dir in patients_root.glob("*") \
                if patient_dir.is_dir() and (patient_dir.name not in self.removed)] \
                    if not patient_dirs else patient_dirs

        self.size = size # Resize images to this size
        self.height, self.width = size 
        self.bbox_expansion = bbox_expansion # Expand bounding boxes by this factor to ensure vertebra is included
        self.bbox_format = bbox_format # Format of bounding boxes, either 'xyxy' or 'xywh'

        resizing    = ia.augmenters.Resize({"height": self.height, "width": self.width}, interpolation='linear')
        resize      = lambda x: resizing.augment_image(x)
        convert     = lambda x: x.astype(np.float32)
        normalize   = lambda x: (x - SuperbStatistics.MEAN) / SuperbStatistics.STD

        self.transforms = [
            resize,
            convert,
            normalize,
            T.ToTensor()
        ]

        self.transforms.extend(transforms)


    def __len__(self) -> int:
        return len(self.patient_dirs)
    
    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
                
            p = Patient.from_moid(self.patient_dirs[index].name, self.patients_root)
            original_height, original_width = p.spine.image.height, p.spine.image.width
            
            image = p.spine.to_numpy()
            for t in self.transforms:
                image = t(image)

            labels  = []
            bboxes  = []
            keypoints = []
            
            for v in p.vertebrae:
                if v.coordinates is not None:

                    # Get label from vertebra
                    labels.append(int(v.grad_visuell) + 1 if v.grad_visuell is not None else 0)

                    # Get keypoint from coordinates
                    keypoint_set = (v.coordinates
                                    .set_size(original_height, original_width)
                                    .resize(self.height, self.width)
                                    .to_numpy())
                    
                    keypoints.append(keypoint_set)

                    bbox = v.coordinates\
                    .to_bbox(original_height, original_width)\
                    .resize(self.height, self.width)

                    bbox = clip_boxes_to_image(
                        box_convert(
                            torch.tensor(bbox.to_expanded(self.bbox_expansion).to_numpy()), 
                            "xywh", self.bbox_format
                        ), 
                        (self.height, self.width))

                    bboxes.append(bbox)

            # Convert to torch tensors
            labels      = torch.tensor(labels, dtype=torch.int64) if len(labels) > 0 else torch.zeros((0), dtype=torch.int64)
            bboxes      = torch.stack(bboxes) if len(bboxes) > 0 else torch.zeros((0,4), dtype=torch.float32)
            keypoints   = torch.tensor(np.array(keypoints), dtype=torch.float32) if len(keypoints) > 0 else torch.zeros((0,3), dtype=torch.float32)

            # Convert to target format
            targets = {
                Target.LABEL.value: labels,
            }
            if Target.BBOX in self.target_format:
                targets[Target.BBOX.value] = bboxes
            
            if Target.KEYPOINT in self.target_format:
                targets[Target.KEYPOINT.value] = keypoints

            return image, targets
    
    def get_patient(self, moid: str) -> Patient:
        return Patient.from_moid(moid, self.patients_root)
    
    def where_label(self, condition: Callable[[Patient], bool]) -> List[Patient]:
        
        patients = []
        for i, patient_dir in enumerate(self.patient_dirs):
            patient = self.get_patient(patient_dir.name)
            if condition(patient):
                patients.append(patient)

        return patients
    
    def filter(self, condition: Callable[[Patient], bool]) -> "SuperbData":
        
        patients = self.where_label(condition)
        patient_dirs = [patient.root / patient.moid for patient in patients]

        ds = SuperbData(
            self.patients_root,
            self.removed,
            self.size,
            patient_dirs,
            self.bbox_expansion,
            self.bbox_format
        )

        return ds

class SuperbDataModule(L.LightningDataModule):

    def __init__(self, 
                 data_dir: Path,
                 batch_size: int = 1,
                 image_size: Tuple[int, int] = (600, 280),
                 target_format: List[Target] = [Target.LABEL, Target.BBOX, Target.KEYPOINT],
                 train_split: float = 0.8,
                 removed: List[str] = [],
                 bbox_expansion: float = 0.1,
                 bbox_format: str = 'xyxy',
                 n_classes: int = 4,
                 n_keypoints: int = 6,
                 n_workers: int = 8,
                 filter: Callable[[Patient], bool] = lambda patient: True
                 ) -> None:
        
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.target_format = target_format
        self.train_split = train_split
        self.bbox_expansion = bbox_expansion
        self.bbox_format = bbox_format
        self.n_workers = n_workers
        self.removed = removed
        self.n_classes = n_classes
        self.n_keypoints = n_keypoints

        self.save_hyperparameters(
            "batch_size",
            "image_size",
            "target_format",
            "train_split",
            "bbox_expansion",
            "bbox_format",
            "n_workers",
            "removed",
            "n_classes",
            "n_keypoints"
        )

        self.data: SuperbData = SuperbData(
            data_dir, 
            size=image_size, 
            removed=removed,
            target_format=target_format,
            bbox_expansion=bbox_expansion,
            bbox_format=bbox_format
            )
        self.filter = filter

    def setup(self, stage: Stage) -> None:

        data = self.data.filter(self.filter)
    
        if stage == Stage.FIT:
            self.train, self.val = random_split(data, [self.train_split, 1 - self.train_split])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train, 
            batch_size=self.batch_size, 
            num_workers=self.n_workers, 
            shuffle=True,
            collate_fn=self.collate,
            pin_memory=True
            )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val, 
            batch_size=self.batch_size, 
            num_workers=self.n_workers, 
            shuffle=False,
            collate_fn=self.collate,
            pin_memory=True
            )
    
    def collate(self, batch):

        images, targets = list(zip(*batch))

        targets = [{k: v for k, v in t.items() if v is not None} for t in targets]

        images = torch.stack(images)

        return images, targets