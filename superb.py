
#%%
import torch
from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd
from typing import *
import preprocessing.labels as labelling
import preprocessing.images as imaging
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision
import json
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import utils
import torch.nn as nn
#%%

class Patchify(nn.Module):

    def __init__(self, resize_shape: Tuple[int, int] , patch_size: int = 256):
        super().__init__()
        self.resize_shape = resize_shape
        self.patch_size = patch_size
        self.resize = torchvision.transforms.Resize(self.resize_shape)

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
    

has_compression = lambda x: \
    int(x.get("GRAD_MORF", 0) or 0) > 0 or \
    int(x.get("GRAD_VISUELL",0) or 0) > 0 or \
    int(x.get("TYP", 0) or 0) > 0

lacks_grad_visuell_or_grad_typ = lambda x: (x.get("GRAD_VISUELL") or x.get("TYP")) and not (x.get("GRAD_VISUELL") and x.get("TYP"))
class SuperbDataset(Dataset):
    
    def __init__(self, 
                 patients_root: Path, 
                 label_type: str = "binary", 
                 transforms: List[nn.Module] = [], 
                 removed: List[str] = [], 
                 class_distribution: Dict[int, int] = {}) -> None:
        
        super().__init__()

        self.removed = removed
        self.patients_root = patients_root
        self.patient_dirs = [patient_dir for patient_dir in patients_root.glob("*") if patient_dir.is_dir() and (patient_dir.name not in self.removed)]
        self.label_type = label_type
        self.class_distribution = class_distribution

        self.transform_list = [
            Normalize(),
            torchvision.transforms.ToTensor(),
        ]
        self.transform_list.extend(transforms)
        self.transforms = torchvision.transforms.Compose(self.transform_list)


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

        image, label, weight = self._read_patient_dir(self.patient_dirs[idx])

        return image, label, weight
    
    def get_idx(self, id: str, label_override=True) -> int:
        
        image, label, weight = self._read_patient_dir(self.patient_dirs[id], label_override=label_override)

        return image, label, weight
    
    def get_patient(self, id: str) -> Tuple[torch.Tensor, torch.Tensor]:
            
        image, label, weight = self._read_patient_dir(self.patients_root / id)
        
        return image, label, weight
    
    def _read_patient_dir(self, patient_dir: Path, label_override: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
         
        patient_id = patient_dir.name
        image_path = patient_dir / "lateral" / (patient_id + ".tiff")
        label_path = patient_dir / "lateral" / (patient_id + ".json")
        image   = cv2.imread(str(image_path), -1)
        weight  = 1

        if self.transforms:
            image = self.transforms(image)

        with open(label_path, "r") as f:
            label = json.load(f)

        if label_override:
            encoded = label
        else:
            if self.label_type == "binary":
                encoded = self._binary_label(label)
                weight  = len(self) / self.class_distribution.get(int(encoded), len(self))
            elif self.label_type == "multilabel":
                encoded = self._categorical_label(label)
            else:
                encoded = label


        
        return image, encoded, weight
    
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
        fig = plt.figure(figsize=(scale*w, scale*h))
        ax = fig.add_subplot(111)

        ax.imshow(image, cmap='gray', origin='lower')
        ax.set_title(label["id"])
        return fig, ax
    
    def _binary_label(self, label: Dict[str, Any]) -> torch.Tensor:
         
        return torch.tensor(int(any([has_compression(label[vertebra]) for vertebra in labelling.VERTEBRA_NAMES]) == True))
    
    def _categorical_label(self, label: Dict[str, Any]) -> torch.Tensor:
         
        vertebra_list = []
        for vertebra in labelling.VERTEBRA_NAMES:
            visual = label[vertebra].get("GRAD_VISUELL") if label[vertebra].get("GRAD_VISUELL") else 0
            type   = label[vertebra].get("TYP") if label[vertebra].get("TYP") else 0
            target  = 1.0 if self.fracture_map[(visual, type)] > 0 else 0.0

            vertebra_list.append(target)

        return torch.tensor(vertebra_list)
