import os
from glob import glob
from pathlib import Path
import pydicom
from typing import *
import utils
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# from torchvision.io import read_image
from PIL import Image
import cv2
class Vertebra:

    def __init__(self, name: str, 
                       number: str, 
                       deformity: str, 
                       severity: str, 
                       user_set_deformity: str, 
                       analysis_type: str, 
                       x: np.ndarray,
                       y: np.ndarray) -> None:

        self.name = name
        self.number = number
        self.deformity = deformity
        self.severity = severity
        self.user_set_deformity = user_set_deformity
        self.analysis_type = analysis_type
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)

class VFA:

    def __init__(self, root: Path) -> None:
        
        self.root = root
        self.d = pydicom.dcmread(list(root.glob("*.dcm"))[0])
        self.pixel_spacing = np.array([float(value) for value in self.d[utils.PIXEL_SPACING_TAG].value])
        self.image = self.d.pixel_array

        annotation = utils.patient_annotation(self.d)

        self.vertebra = {k: Vertebra(
            name=k,
            number = annotation[k]["NUMBER"],
            deformity = annotation[k]["DEFORMITY"],
            severity = annotation[k]["SEVERITY"],
            user_set_deformity = annotation[k]["USER_SET_DEFORMITY"],
            analysis_type = annotation[k]["ANALYSIS_TYPE"],
            x=np.array(annotation[k]["x"]),
            y=np.array(annotation[k]["y"]),
        ) for k in annotation.keys()}


    def plot(self, scale=1.2, offset=[100,0]) -> Tuple[plt.figure, plt.Axes]:

        w, h = matplotlib.figure.figaspect(self.image)
        fig = plt.figure(figsize=(scale*w, scale*h))
        ax = fig.add_subplot(111)

        ax.imshow(np.flip(self.image, 0), cmap='gray', origin='lower')

        for vertebra in self.vertebra:

            ax.scatter(vertebra.x/self.pixel_spacing[0], 
                       vertebra.y/self.pixel_spacing[1], 
                       marker='x', c='r')
            ax.text(vertebra.x[0]/self.pixel_spacing[0]+offset[0], 
                    vertebra.y[0]/self.pixel_spacing[1]+offset[1], 
                    vertebra.name, 
                    c='r')

        return fig, ax

    def __str__(self) -> str:
        return ", ".join([v.name for v in self.vertebra])

    def __repr__(self) -> str:
        return str(self)

class Reports:

    def __init__(self, root: Path) -> None:
        self.ds = [pydicom.dcmread(path) for path in root.glob("*.dcm")]
        self.images = [d.pixel_array for d in self.ds]

class Patient:

    def __init__(self, id: str, root: Path | str, vfa: bool = False, reports: bool = False) -> None:

        if isinstance(root, str):
            root = Path(root)

        self.root = root
        self.id = id
        self.vfa = VFA(root / "dicom" / "vfa") if (root / "dicom" / "vfa").exists() and vfa else None
        self.reports = Reports(root / "dicom" / "reports") if (root / "dicom" / "reports").exists and reports else None

    @property
    def image(self) -> np.ndarray:
        filename = self.id + ".tiff"
        file_path = self.root / self.id / "dicom" / "vfa" / "image" / filename
        image = cv2.imread(str(file_path), -1) if file_path.exists() else None

        return image

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        return str(self)
    
    @staticmethod
    def has_image(id: str, root: Path | str) -> bool:
        if isinstance(root, str):
            root = Path(root)
        filename = id + ".tiff"
        file_path = root / id / "dicom" / "vfa" / "image" / filename
        return file_path.exists()

class PatientCollection(dict):
    """
    Creates an iterable collection of patients as a dictionary
    
    """

    def __init__(self, root: Path | str, vfa: bool = False, reports: bool = False) -> None:

        if isinstance(root, str):
            root = Path(root)
        
        self.root = root

        files = list(root.glob("*/"))

        self.update({
            file.name: Patient(file.name, file, vfa, reports) for file in files
        })

def filter_patients(f, ps: PatientCollection):

    return list(filter(f, ps.values()))
