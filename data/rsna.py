from pathlib import Path
import pydicom
from typing import *
import torch
from torchvision.transforms import transforms as T
from torch.utils.data import Dataset
import pandas as pd

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