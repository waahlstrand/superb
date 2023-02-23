
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
from torchvision import transforms
import json
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import utils
#%%

has_compression = lambda x: x["GRAD_MORF"] != 0 or x["GRAD_VISUELL"] != 0 or x["TYP"] != 0

class SuperbDataset(Dataset):
    
    def __init__(self, patients_root: Path, label_type: str = "binary") -> None:
        super().__init__()

        self.patients_root = patients_root
        self.patient_dirs = [patient_dir for patient_dir in patients_root.glob("*") if patient_dir.is_dir()]
        self.label_type = label_type
        # First index: GRAD_VISUELL
        # 1: mild compression
        # 2: moderate compression
        # 3: severe compression
        # Second index: TYP
        # 1: wedge
        # 2: concave
        # 3: crush
        self.fracture_map = {
            (None, None): -1, # no compression
            (0, None): -1, # no compression
            (None, 0): -1, # no compression
            (0,0): -1, # no compression
            (1,1): 0, # mild wedge
            (1,2): 1, # mild concave
            (1,3): 2, # mild crush
            (2,1): 3, # moderate wedge
            (2,2): 4, # moderate concave
            (2,3): 5, # moderate crush
            (3,1): 6, # severe wedge
            (3,2): 7, # severe concave
            (3,3): 8, # severe crush

        }

    def __len__(self) -> int:

        return len(self.patient_dirs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        image, label = self._read_patient_dir(self.patient_dirs[idx])

        return image, label
    
    def get_patient(self, id: str) -> Tuple[torch.Tensor, torch.Tensor]:
            
        image, label = self._read_patient_dir(self.patients_root / id)
        
        return image, label
    
    def _read_patient_dir(self, patient_dir: Path, label_override: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
         
        patient_id = patient_dir.name
        image_path = patient_dir / "lateral" / (patient_id + ".tiff")
        label_path = patient_dir / "lateral" / (patient_id + ".json")

        image   = cv2.imread(str(image_path), -1)
        with open(label_path, "r") as f:
            label = json.load(f)

        if self.label_type == "binary":
            encoded = self._binary_label(label)
        elif self.label_type == "categorical":
            encoded = self._categorical_label(label)
        else:
            encoded = label

        if label_override:
            encoded = label
        
        return image, encoded
    
    def visualize_item(self, idx, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
            
        image, label = self._read_patient_dir(self.patient_dirs[idx], label_override=True)
        
        f, ax = self.plot(image, label, **kwargs)

        return f, ax
    
    def visualize_patient(self, id, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
            
        image, label = self._read_patient_dir(self.patients_root / id, label_override=True)
        
        f, ax = self.plot(image, label, **kwargs)

        return f, ax
    
    def plot(self, image, label, scale=1.2, offset=[200,50]) -> Tuple[plt.Figure, plt.Axes]:

        w, h = matplotlib.figure.figaspect(image)
        fig = plt.figure(figsize=(scale*w, scale*h))
        ax = fig.add_subplot(111)

        ax.imshow(np.flip(image, 0), cmap='gray', origin='lower')

        for i, vertebra in enumerate(labelling.VERTEBRA_NAMES):

            if vertebra in label:
                if "x" in label[vertebra] and "y" in label[vertebra]:

                    c = "r" if i % 2 else "m"

                    x = np.array(label[vertebra]["x"])
                    y = np.array(label[vertebra]["y"])

                    ax.scatter(x/label["pixel_spacing"][0], 
                            y/label["pixel_spacing"][1], 
                            marker='x', c=c)
                    ax.text(x[0]/label["pixel_spacing"][0]+offset[0], 
                            y[0]/label["pixel_spacing"][1]+offset[1], 
                            vertebra + " " +  "(" + str(self.fracture_map[(label[vertebra].get("GRAD_VISUELL", 0), label[vertebra].get("TYP", 0))])+")",
                            c=c)

        return fig, ax
    
    def _binary_label(self, label: Dict[str, Any]) -> torch.Tensor:
         
        return torch.tensor(int(any([has_compression(label[vertebra]) for vertebra in labelling.VERTEBRA_NAMES]) == True))
    
    def _categorical_label(self, label: Dict[str, Any]) -> torch.Tensor:
         
        return torch.tensor([
            self.fracture_map[(label[vertebra].get("GRAD_VISUELL", 0), label[vertebra].get("TYP", 0))] 
            for vertebra in labelling.VERTEBRA_NAMES
            ])
