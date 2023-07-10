from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.ops import box_convert, clip_boxes_to_image
import torch
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
from typing import *
from utils.typings import *

class Superb(Dataset):

    def __init__(self, 
                 patients_root: Path,
                 removed: List[str] = [],
                 dtype: np.dtype = np.float32,
                 size: Tuple[float, float] = (600, 280),
                 patient_dirs: List[Path] = [],
                 severity: int = 0,
                 bbox_expansion: float = 0.1,
                 mode: str = "exists"
                 ) -> None:
        super().__init__()

        self.patients_root = patients_root
        self.removed = removed
        self.patient_dirs = [
            patient_dir for patient_dir in patients_root.glob("*") \
                if patient_dir.is_dir() and (patient_dir.name not in self.removed)] \
                    if not patient_dirs else patient_dirs

        self.dtype = dtype
        self.mode = mode
        self.height, self.width = size
        self.bbox_expansion = bbox_expansion

        self.compression = Compression(severity, mode)

    def __len__(self) -> int:
        return len(self.patient_dirs)
    
    def __getitem__(self, index) -> Patient:
            
            patient_dir = self.patient_dirs[index]
    
            return Patient.from_moid(patient_dir.name, self.patients_root)
    
    def __iter__(self) -> Iterator[Patient]:

        for patient_dir in self.patient_dirs:
            yield Patient.from_moid(patient_dir.name, self.patients_root)
    
    def get_patient(self, moid: str) -> Patient:
        return Patient.from_moid(moid, self.patients_root)
    
    def where_label(self, condition: Callable[[Patient], bool]) -> List[int]:
        
        idxs = []
        for i, patient in enumerate(self):
            if condition(patient):
                idxs.append(i)

        return idxs
    
    def filter(self, condition: Callable[[Patient], bool]) -> None:
        
        idxs = self.where_label(condition)
        self.patient_dirs = [self.patient_dirs[idx] for idx in idxs]

        return self
    
def collate(ds: Superb, batch: List[Patient]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collate function for the Superb dataset. Returns a tuple of images and labels.

    Args:
        ds: The dataset.
        batch: The batch of patients.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of images and labels.
    """
        
    images = [patient.spine.to_numpy() for patient in batch]
    labels = [np.array([ds.compression(v) for v in patient.vertebrae]) for patient in batch]
        
    return np.array(images), np.array(labels)

def collate_with_coords(ds: Superb, batch: List[Patient]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collate function for the Superb dataset. Returns a tuple of images, labels and coordinates.

    Args:
        ds: The dataset.
        batch: The batch of patients.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple of images, labels and coordinates.
    """
        
    images = [patient.spine.to_numpy(ds.height, ds.width, dtype=ds.dtype) for patient in batch]
    labels = [np.array([ds.compression(v) for v in patient.vertebrae]) for patient in batch]
    coords = [patient.vertebrae.coordinates_to_numpy() for patient in batch]
        
    return np.array(images), np.array(labels), np.array(coords)

def collate_with_bboxes(ds: Superb, batch: List[Patient]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collate function for the Superb dataset. Returns a tuple of images, labels and bounding boxes.

    Args:
        ds: The dataset.
        batch: The batch of patients.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple of images, labels and bounding boxes.
    """
    
    images  = torch.stack([patient.spine.to_numpy(ds.height, ds.width, ds.dtype) for patient in batch])

    boxes = []
    keypoints = []
    labels = []
    for patient in batch:
        bs = []
        ks = []
        ls = []
        height, width = patient.spine.image.height, patient.spine.image.width
        for v in patient.vertebrae:
            if v.coordinates is not None:
                
                # Get label
                compression = ds.compression(v) + 1 # 0 is background
                ls.append(torch.tensor(compression, dtype=torch.int64))

                # Get bounding box
                box = v.coordinates.to_bbox(height, width).resize(ds.height, ds.width)
                box = clip_boxes_to_image(box_convert(torch.tensor(box.to_expanded(ds.bbox_expansion).to_numpy()), "xywh", "xyxy"), (ds.height, ds.width))
                bs.append(box)

                # Get keypoints
                # print(height, width, ds.height, ds.width)
                coordinates = v.coordinates.set_size(height, width).resize(ds.height, ds.width).to_numpy()
                
                k = torch.tensor(coordinates)
                ks.append(k)

        labels.append(ls)
        keypoints.append(ks)
        boxes.append(bs)

    targets = [
        {
            "labels": torch.stack(ls) if len(ls) > 0 else torch.zeros((0,1), dtype=torch.int64),
            "boxes": torch.stack(bs) if len(bs) > 0 else torch.zeros((0,4), dtype=torch.float32),
            "keypoints": torch.stack(ks).float() if len(ks) > 0 else torch.zeros((0,2), dtype=torch.float32)
        }
        for ls, bs, ks in zip(labels, boxes, keypoints)
                  ]
    
    return images, targets


class FullSpineDataset(Dataset):

    def __init__(self, 
                 patients_root: Path,
                 removed: List[str] = [],
                 dtype: np.dtype = np.float32,
                 size: Tuple[float, float] = (600, 280),
                 patient_dirs: List[Path] = [],
                 severity: int = 0,
                 bbox_expansion: float = 0.1,
                 mode: str = "exists"
                 ) -> None:
        super().__init__()

        self.patients_root = patients_root
        self.removed = removed
        self.patient_dirs = [
            patient_dir for patient_dir in patients_root.glob("*") \
                if patient_dir.is_dir() and (patient_dir.name not in self.removed)] \
                    if not patient_dirs else patient_dirs
        
        self.patients = [Patient.from_moid(patient_dir.name, self.patients_root) for patient_dir in self.patient_dirs]
        
        self.vertebrae = [(patient, v) for patient in self.patients for v in patient.vertebrae if v.coordinates is not None]


        self.dtype = dtype
        self.mode = mode
        self.height, self.width = size
        self.bbox_expansion = bbox_expansion

        self.compression = Compression(severity, mode)

    def __len__(self) -> int:
        return len(self.vertebrae)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        p, v = self.vertebrae[index]

        coordinates = v.coordinates.to_bbox(*p.spine.size).to_xcycwh()

        # Centre coordinates
        centre = coordinates[:2]
        image = p.spine.crop(v.coordinates.to_bbox(*p.spine.size).to_expanded(0.4))

        image = torch.Tensor(image)
        centre = torch.Tensor(centre)
        label = torch.tensor(self.compression(v)-1, dtype=torch.int64)

        return image, centre, label




