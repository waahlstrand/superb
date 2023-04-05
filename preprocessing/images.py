from patients import Patient
import pydicom
import numpy as np

MIN_WIDTH = 567
MAX_WIDTH = 603

MIN_HEIGHT = 1248
MAX_HEIGHT = 1656

NORMALIZATION = (1000, 4000) # Approximate maximum pixel value of all images
SHAPE = (512, 1024)
# PADDING_SHAPE = (600, 1800)
PADDING_SHAPE = (1800, 600)


normalize = lambda x: (np.clip(x, NORMALIZATION[0], NORMALIZATION[1]).astype(np.float64) - NORMALIZATION[0]) / (NORMALIZATION[1] - NORMALIZATION[0])

def normalized_patient_image(p: Patient, window=False) -> np.ndarray:

    # Perform contrast sharpening from DICOM window
    if window:
        windowed = pydicom.pixel_data_handlers.apply_windowing(p.image, p.vfa.d)
    else:
        windowed = p.image

    # Max-normalization to float
    image = normalize(windowed)

    return image

def heuristic_height(vertebra: int, n_vertebrae: int):
    """
    Heuristic for approximately locating the height of the vertebrae.
    """
    normalized_vertebra = vertebra / n_vertebrae

    centres = np.linspace(
        start=1/(n_vertebrae+1), 
        stop=n_vertebrae/(n_vertebrae+1), 
        num=n_vertebrae)
    
