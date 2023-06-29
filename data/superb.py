from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.ops import box_convert, clip_boxes_to_image
import torch
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
from typing import *
from typings import *

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

    labels = [[torch.tensor(ds.compression(v)-1, dtype=torch.int64) for v in patient.vertebrae if ds.compression(v)] for patient in batch]
    # boxes  = [[torch.tensor(v.coordinates.to_bbox().resize(ds.height, ds.width).to_x1y1x2y2()) for v in patient.vertebrae if ds.compression(v) and v.coordinates is not None] for patient in batch]            
    boxes = []
    for patient in batch:
        bs = []
        for v in patient.vertebrae:
            if ds.compression(v) and v.coordinates is not None:
                box = v.coordinates.to_bbox(patient.spine.image.height, patient.spine.image.width).resize(ds.height, ds.width)
                box = clip_boxes_to_image(box_convert(torch.tensor(box.to_expanded(ds.bbox_expansion).to_numpy()), "xywh", "xyxy"), (ds.height, ds.width))
                bs.append(torch.tensor(box))
        boxes.append(bs)
    # print(boxes)
    # boxes  = [[
        
    #     box_convert(torch.tensor(
    #     v.coordinates\
    #         .to_bbox(patient.spine.image.height, patient.spine.image.width)\
    #         .resize(ds.height, ds.width)\
    #         .to_numpy()), "xywh", "xyxy" ) for v in patient.vertebrae if ds.compression(v) and v.coordinates is not None] for patient in batch]            


    targets = [
        {
            "labels": torch.stack(ls) if len(ls) > 0 else torch.zeros((0,1), dtype=torch.int64),
            "boxes": torch.stack(bs) if len(bs) > 0 else torch.zeros((0,4), dtype=torch.float32),
        }
        for ls, bs in zip(labels, boxes)
                  ]
    
    return images, targets


class Vertebrae(Dataset):

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




class RSNA(Dataset):

    def __init__(self, image_root: Path, labels_path: Path, size=(1024, 1024)) -> None:
        super().__init__()

        self.image_root = image_root
        self.labels_path = labels_path
        self.size = size
        self.labels = self._format_labels(pd.read_csv(labels_path))
        self.files  = list(self.image_root.glob("*.dcm"))

        self.resize = T.Resize(size)

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> torch.Tensor:

        file        = self.files[index]
        patient_id  = file.stem
        image       = pydicom.dcmread(file).pixel_array

        # Normalize
        image       = torch.from_numpy(image).float() / 255

        # Repeat channels
        # image       = image.repeat(3, 1, 1)
        image = image.unsqueeze(0)

        # Resize
        image       = self.resize(image)

        label       = self.get_label(file)

        return image, label
    
    def get_label(self, file: Path) -> int:

        return self.labels[file.stem]["Target"]
    
    def _format_labels(self, labels: pd.DataFrame):

        return labels[["patientId", "Target"]].groupby("patientId").agg("mean").to_dict(orient="index")


    def where_label(self, condition: Callable[[int], bool]) -> List[int]:
        
        labels = []
        for i, file in enumerate(self.files):

            label = self.get_label(file)

            if condition(label):
                labels.append(i)

        return labels