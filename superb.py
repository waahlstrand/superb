
#%%
import torch
from pathlib import Path
from torch.utils.data import Dataset
from patients import Patient
import pandas as pd
from typing import *
import preprocessing.labels as labelling
import preprocessing.images as imaging
from tqdm import tqdm
import shutil
import os
import matplotlib.pyplot as plt
from torchvision import transforms
import json
#%%

class SuperbDataset(Dataset):
    
    def __init__(self, labels_path: Path, patients_root: Path, label_type: str = "binary", vfa: bool = True) -> None:
        super().__init__()

        self.image2excel, self.excel2image = labelling.build_conversion_dicts(patients_root)
        self.patients_root  = patients_root
        self.df = pd.read_excel(labels_path)
        labels = labelling.excel_to_records(self.df)
        self.labels = self._only_vfa_records(labels)
        self.label_type = label_type
        self.vfa = vfa

    def __len__(self) -> int:

        return len(self.labels)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:

        patient_id = self.excel2image[self.labels[idx]["id"]]
        patient = Patient(patient_id, self.patients_root, vfa=self.vfa)
        image   = imaging.normalized_patient_image(patient)
        # image   = transforms.CenterCrop((512, 1024))(image)

        if self.label_type == "binary":
            label = self._binary_label(self.labels[idx])
        else:
            label = self.labels[idx]
        
        return image, label
    
    def get_patient(self, idx, source = "excel", **kwargs) -> Patient:
            
            if source == "excel":
                patient_id = self.excel2image[idx]
            elif source == "image":
                 patient_id = idx
            else:
                raise ValueError("Source must be either 'excel' or 'image'")
            
            patient = Patient(patient_id, self.patients_root, **kwargs)
            
            return patient
    
    def visualize_item(self, idx, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
            
            image, label = self.__getitem__(idx)
            f, ax = plt.subplots()
            ax.imshow(image, cmap="gray")

            return f, ax
    
    def visualize_patient(self, idx, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
            
            patient = self.get_patient(idx, **kwargs)

            if self.vfa:
                f, ax = patient.vfa.plot()
                        
            else:
                f, ax = plt.subplots()
                ax.imshow(patient.image, cmap="gray")

            return f, ax
    
    def _binary_label(self, label: Dict[str, Any]) -> torch.Tensor:
         
        has_compression = lambda x: x["GRAD_MORF"] != 0 or x["GRAD_VISUELL"] != 0 or x["TYP"] != 0
         
        return torch.tensor(int(any([has_compression(label[vertebra]) for vertebra in labelling.VERTEBRA_NAMES]) == True))
    
    def _only_vfa_records(self, labels) -> List[Dict[str, Any]]:
            
            return [record for record in labels if Patient.has_image(record["id"], self.patients_root)]
         
