from patients import Patient
import pydicom
import numpy as np

MIN_WIDTH = 567
MAX_WIDTH = 603

MIN_HEIGHT = 1248
MAX_HEIGHT = 1656

NORMALIZATION = 4500 # Approximate maximum pixel value of all images
SHAPE = (512, 1024)
PADDING_SHAPE = (1800, 600)

normalize = lambda x: x / NORMALIZATION

def normalized_patient_image(p: Patient, window=False) -> np.ndarray:

    # Perform contrast sharpening from DICOM window
    if window:
        windowed = pydicom.pixel_data_handlers.apply_windowing(p.image, p.vfa.d)
    else:
        windowed = p.image

    # Max-normalization to float
    image = normalize(windowed)

    return image
