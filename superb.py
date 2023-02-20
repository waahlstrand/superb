
#%%
import torch
from pathlib import Path
from torch.utils.data import Dataset
from patients import Patient
import pandas as pd
from typing import *
import preprocessing.labels as labelling
import preprocessing.images as imaging

#%%

class SuperbDataset(Dataset):
    
    def __init__(self, labels_path: Path, patients_root: Path) -> None:
        super().__init__()

        self.image2excel, self.excel2image = labelling.build_conversion_dicts(patients_root)
        self.patients_root  = patients_root
        self.df = pd.read_excel(labels_path)
        labels = labelling.excel_to_records(self.df)
        self.labels = self._only_vfa_records(labels)

    def __len__(self) -> int:

        return len(self.labels)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:

        patient_id = self.excel2image[self.labels[idx]["id"]]
        patient = Patient(patient_id, self.patients_root, vfa=True)
        image   = imaging.normalized_patient_image(patient)
        
        return image, self.labels[idx]
    
    def get_patient(self, idx, source = "excel") -> Patient:
            
            if source == "excel":
                patient_id = self.excel2image[self.df[idx]["ID"]]
            elif source == "image":
                 patient_id = idx
            else:
                raise ValueError("Source must be either 'excel' or 'image'")
            
            patient = Patient(patient_id, self.patients_root)
            
            return patient
    
    def _only_vfa_records(self, labels) -> List[Dict[str, Any]]:
            
            return [record for record in labels if Patient.has_image(record["id"], self.patients_root)]
         
    
#%%
# d       = labelling.excel_to_dict(df)
# records = labelling.excel_to_records(df)
# vf      = labelling.excel_to_vertebra_dataframe(df)

ds = SuperbDataset(Path("data/labels.xlsx"), Path("data/patients"))