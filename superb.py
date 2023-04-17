import torch
from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd
from typing import *
import preprocessing.labels as labelling
import preprocessing.images as imaging
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms as T
import json
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import utils
import torch.nn as nn
from augmentations import Preprocess
from kornia import augmentation as K

has_compression = lambda x: \
    int(x.get("GRAD_MORF", 0) or 0) > 0 or \
    int(x.get("GRAD_VISUELL",0) or 0) > 0 or \
    int(x.get("TYP", 0) or 0) > 0



class Compression:

    def __init__(self, severity: int, mode: str = "exists") -> None:
        
        self.severity = severity
        self.mode = mode

    def __call__(self, x: Dict[str, str]) -> bool:

        grad_morf    = x.get("GRAD_MORF")
        grad_visuell = x.get("GRAD_VISUELL")
        typ          = x.get("TYP")

        if grad_morf is None:
            grad_morf = 0
        else:
            grad_morf = float(grad_morf)
        
        if grad_visuell is None:
            grad_visuell = 0
        else:
            grad_visuell = int(grad_visuell)

        if typ is None:
            typ = 0
        else:
            typ = int(typ)

        has_compression = all([grad_morf > 0, grad_visuell > self.severity, typ > 0])

        if has_compression and any([grad_morf == 0, grad_visuell == self.severity, typ == 0]):
            raise ValueError(f"Inconsistencies in compression labels. {x}")
        
        if self.mode == "exists":
            return has_compression
        elif self.mode == "severity":
            return grad_visuell
        else:
            raise ValueError(f"Unknown compression mode: {self.mode}")


class SuperbDataset(Dataset):
    
    def __init__(self, 
                 patients_root: Path, 
                 target_size: Tuple[int, int] = imaging.PADDING_SHAPE,
                 removed: List[str] = [], 
                 class_distribution: Dict[int, int] = {}) -> None:
        
        super().__init__()

        self.removed = removed
        self.patients_root = patients_root
        self.patient_dirs = [patient_dir for patient_dir in patients_root.glob("*") if patient_dir.is_dir() and (patient_dir.name not in self.removed)]
        self.target_size = target_size
        self.class_distribution = class_distribution
        self.preprocess = Preprocess(target_size=self.target_size)

        # First index: GRAD_VISUELL
        # 1: mild compression
        # 2: moderate compression
        # 3: severe compression
        # Second index: TYP
        # 1: wedge
        # 2: concave
        # 3: crush
        self.fracture_map = {
            (None, None): 0, # no compression
            (0, None): 0, # no compression
            (None, 0): 0, # no compression
            (0,0): 0, # no compression
            (1,1): 1, # mild wedge
            (1,2): 2, # mild concave
            (1,3): 3, # mild crush
            (2,1): 4, # moderate wedge
            (2,2): 5, # moderate concave
            (2,3): 6, # moderate crush
            (3,1): 7, # severe wedge
            (3,2): 8, # severe concave
            (3,3): 9, # severe crush
            # (0,2): 1,

        }

    def __len__(self) -> int:

        return len(self.patient_dirs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        return self._read_patient_dir(self.patient_dirs[idx])
    
    def get_idx(self, id: str, label_override=True) -> Tuple[torch.Tensor, torch.Tensor]:
        
        return self._read_patient_dir(self.patient_dirs[id], label_override=label_override)
    
    def get_patient(self, id: str) -> Tuple[torch.Tensor, torch.Tensor]:
            
        return self._read_patient_dir(self.patients_root / id)
    
    def get_raw(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        return self._image_label_from_dir(self.patient_dirs[idx], label_override=False)

    def _image_label_from_dir(self, patient_dir: Path, label_override: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        
        patient_id = patient_dir.name
        image_path = patient_dir / "lateral" / (patient_id + ".tiff")
        label_path = patient_dir / "lateral" / (patient_id + ".json")

        assert image_path.exists(), f"Image path does not exist: {image_path}"
        assert label_path.exists(), f"Label path does not exist: {label_path}"

        # Read image
        image = self.preprocess(str(image_path))

        # Get label
        if not label_override:
            label = self.label_from_path(label_path)
        else:
            with open(label_path, "r") as f:
                label = json.load(f)
        
        return image, label
            
    def _read_patient_dir(self, patient_dir: Path, label_override: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
         
        # Read image and label
        image, label = self._image_label_from_dir(patient_dir, label_override=label_override)
        
        # Get weight
        weight = self.weight_from_class_distribution(label)

        return image, label, weight
    
    def visualize_item(self, idx, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
            
        image, label, weight = self._read_patient_dir(self.patient_dirs[idx], label_override=True)
        
        f, ax = self.plot(image, label, **kwargs)

        return f, ax
    
    def visualize_patient(self, id, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
            
        image, label, weight = self._read_patient_dir(self.patients_root / id, label_override=True)
        
        f, ax = self.plot(image, label, **kwargs)

        return f, ax
    
    def plot(self, image, label, scale=1.2, offset=[200,50]) -> Tuple[plt.Figure, plt.Axes]:

        w, h = matplotlib.figure.figaspect(image)
        fig, ax = plt.subplots(figsize=(scale*w, scale*h))

        ax.imshow(image, cmap='gray')
        ax.set_title(label["id"])

        return fig, ax
    
    def label_from_path(self, path: Union[str, Path]) -> torch.Tensor:

        # Read label metadata
        with open(path, "r") as f:
            metadata = json.load(f)
        
        return metadata
    
    def weight_from_class_distribution(self, label: Dict[str, Any]) -> torch.Tensor:

        return 1.0
    
    def get_class_distribution(self) -> Dict[str, int]:
        
        return {}
    

class BinaryDataset(SuperbDataset):

    def __init__(self, *args, severity: int = 0, mode: str = "exists", **kwargs) -> None:
        
        super().__init__(*args, **kwargs)

        self.severity = severity 
        self.mode = mode
        self.has_compression = Compression(severity=self.severity, mode=self.mode)
        self.class_distribution = self.get_class_distribution()

    def label_from_path(self, path: str) -> torch.Tensor:

        # Read label metadata
        with open(path, "r") as f:
            metadata = json.load(f)

        if self.mode == "exists":
            return torch.tensor(int(any([self.has_compression(metadata[vertebra]) for vertebra in labelling.VERTEBRA_NAMES]) == True))
        elif self.mode == "severity":
            return torch.tensor(int(max([self.has_compression(metadata[vertebra]) for vertebra in labelling.VERTEBRA_NAMES])))
        else:
            raise ValueError(f"Mode {self.mode} not supported.")
        
    
    def weight_from_class_distribution(self, y: torch.Tensor) -> torch.Tensor:

        # Convert to binary
        label = torch.minimum(y, torch.tensor(1)).item()

        # Get weight for class imbalance
        if self.class_distribution:
            weight = ( self.class_distribution[label] ) / len(self)
            
            return 1.0
        else:
            return 1.0
        
    def get_class_distribution(self) -> Dict[str, int]:
        
        class_distribution = {}
        for idx in tqdm(range(len(self)), desc="Calculating class distribution..."):

            image, label = self.get_raw(idx)

            # Convert to binary
            label = torch.minimum(label, torch.tensor(1)).item()

            if label not in class_distribution:

                class_distribution[label] = 0

            class_distribution[label] += 1

        print("\n Class distribution:", class_distribution, "\n")
        
        return class_distribution
    
class CategoricalDataset(SuperbDataset):

    def label_from_path(self, path: str) -> torch.Tensor:
        
        # Read label metadata
        with open(path, "r") as f:
            metadata = json.load(f)

        vertebra_list = []
        for vertebra in labelling.VERTEBRA_NAMES:
            visual = metadata[vertebra].get("GRAD_VISUELL") if metadata[vertebra].get("GRAD_VISUELL") else 0
            type   = metadata[vertebra].get("TYP") if metadata[vertebra].get("TYP") else 0
            target  = 1.0 if self.fracture_map[(visual, type)] > 0 else 0.0

            vertebra_list.append(target)

        return torch.tensor(vertebra_list)