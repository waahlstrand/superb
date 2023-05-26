#%%
import os
from glob import glob
from tqdm import tqdm
from lxml import etree
from typing import *
import numpy as np
import pydicom
from utils.xml_to_dict import XmlDictConfig
import utils.labels as labelling
import shutil
from pathlib import Path
from PIL import Image
import pandas as pd
import json 
from utils.parse import parse_annotation_pdf, vertebrae_from_points
import itk

ANNOTATION_TAG = (0x0021, 0x1001)
PATIENT_TAG = (0x0010, 0x0020)
PIXEL_SPACING_TAG = (0x0028, 0x0030)


def threshold(img: np.ndarray, window_width: float, window_level: float) -> np.ndarray:
    """Thresholds an image based on a window width and window center.
    Args:
        image (np.ndarray): The image to threshold.
        window_width (float): The window width.
        window_center (float): The window center.
    Returns:
        np.ndarray: The thresholded image.
    """
    image = img.copy()
    image = image.squeeze()
    # image = image - window_level
    max_value = window_level + window_width / 2
    min_value = window_level - window_width / 2
    
    image[image < min_value] = min_value
    image[image > max_value] = max_value

    return image

def transform_coordinates_from_pdf_to_image(
        coords: List[List[float]], 
        true_height: int, 
        true_width: int, 
        coord_height: int, 
        coord_width: int) -> List[List[float]]:
    """
    Transform coordinates from PDF to image coordinates.
    """
    # Scale the coordinates
    coords = [[x * true_width / coord_width, y * true_height / coord_height] for y, x in coords]

    # Flip the coordinates
    # coords = [[x, true_height - y] for x, y in coords]

    return coords


def get_patient_pdfs(path: Path) -> Dict[str, Union[Path, None]]:

    pdfs = list(path.rglob("*.pdf"))

    patient_pdfs = {}
    for pdf in pdfs:
        moid = labelling.pdf_name_to_moid(pdf)
        patient_pdfs.update({moid: pdf})

    return patient_pdfs


def build_patient_directory_tree(
        dcm_root: Path, 
        pdf_root: Path,
        labels_path: Path, 
        target_root: Path):
    """
    Build a directory tree for the patient data.
    
    Structure:
        - patient_id
            - ct
                - leg
                - arm
            - lateral
                - patient_id.tiff
                - patient_id.json
                - patient_id.dcm
            - reports
                - [...].dcm

    Args:
        dcm_root (Path): The root directory containing the DICOM files.
        labels_path (Path): The path to the labels Excel file.
        target_root (Path): The root directory to which the directory tree will be built.
    """

    files   = list(dcm_root.glob("*.dcm"))
    labels  = pd.read_excel(labels_path)
    pdfs    = get_patient_pdfs(pdf_root)

    is_vfa_dicom = lambda x: x[PIXEL_SPACING_TAG].value is not None
    has_pdf_annotation = lambda x: x in pdfs.keys()
    has_full_annotation = lambda vertebrae, names: len(vertebrae) == len(names)

    # Group files by patient
    lacks_vfa = []
    errors = []

    # Get labels
    labels = labelling.excel_to_dict(labels)

    patient_has_vfa = {}
    not_in_excel = []
    progress = tqdm(files, desc="Iterating over patient dicom files")
    for file in progress:

        has_vfa = False

        # Open the dicom files
        d = pydicom.dcmread(file)

        # Extract the patient ID
        patient_id = labelling.image_id_to_excel_id(d.PatientID)

        progress.set_description(f"Processing patient {patient_id}")

        # Check if the patient already has a directory
        lateral_file_dir =  target_root / patient_id / "lateral" 
        report_file_dir  = target_root / patient_id / "reports"

        if not os.path.exists(lateral_file_dir):
            os.makedirs(lateral_file_dir)

        if not os.path.exists(report_file_dir):
            os.makedirs(report_file_dir)

        # If the file is a VFA dicom file
        #  - Create a renamed dicom file
        #  - Download the image as an easy access tiff
        #  - Get the label for the image
        #  - Check if there is a PDF annotation
        #  - Get the annotation from the PDF
        #  - Save all the files

        if is_vfa_dicom(d):
            has_vfa = True

            # Create a renamed dicom file
            new_dcm_file    = lateral_file_dir / (patient_id + ".dcm")

            # Download the image as an easy access tiff
            new_image_file  = lateral_file_dir / (patient_id + ".tiff")

            # Get the label for the image
            new_label_file  = lateral_file_dir / (patient_id + ".json")

            # Save all the files
            ## Save the dicom file
            d.save_as(new_dcm_file)

            ## Easiest with PIL
            img = Image.fromarray(d.pixel_array)
            img.save(new_image_file)

            height, width = d.pixel_array.shape

            ## Handle the label information sources
            dx, dy     = d.PixelSpacing
            try:
                label      = labels[patient_id]
            except KeyError:
                not_in_excel.append(patient_id)
                continue

            label.update({
                "moid": patient_id,
                "pixel_spacing": [dx, dy],
                "height": height,
                "width": width,
                })

            # Get the annotation from the PDF
            if has_pdf_annotation(patient_id):

                pdf = pdfs[patient_id]

                # Get the annotation from the PDF
                try:
                    annotation, coord_width, coord_height, names = parse_annotation_pdf(pdf)
                except Exception as e:
                    errors.append({"patient_id": patient_id, "error": str(e)})
                    print(e, pdf)
                    continue

                # Get the vertebrae from the annotation
                vertebrae = vertebrae_from_points(
                    annotation, 
                    names,
                    n_points_in_vertebra=6, 
                    n_neighbours=8, 
                    area_threshold=3000, 
                    rectangularity_threshold=.5, 
                    height_width_ratio_threshold=0.85)

                # Add the vertebrae to the label
                if has_full_annotation(vertebrae, names):
                    for name in names:
                        if name in vertebrae.keys():
                            label[name].update({"coordinates": vertebrae[name]})
                        else:
                            try:
                                label.update({name: {"coordinates": vertebrae[name]}})
                            except KeyError:

                                raise KeyError

                    label.update({
                        "coord_height": coord_height,
                        "coord_width": coord_width,
                        })

            with open(new_label_file, "w+") as f:
                json.dump(label, f, indent=4)
                
        # If the file is not a VFA dicom file, it is a report
        else:
            # Create a renamed dicom file
            new_dcm_file = report_file_dir / file.name

            # Save the dicom file
            d.save_as(new_dcm_file)

        # Release memory of dicom file
        del d
        
        if patient_id not in patient_has_vfa.keys():
            patient_has_vfa[patient_id] = []

        patient_has_vfa[patient_id].append(has_vfa)

    return patient_has_vfa, not_in_excel, errors


def group_cts(root: Path):

    cts = list(root.rglob("*.ISQ:1"))
    patients: Dict[str, List[Path]] = {}

    for ct in cts:

        data = itk.imread(str(ct))
        metadata = dict(data)

        patient_str, measurement = metadata["PatientName"].split(" ")
        patient_id = labelling.image_id_to_excel_id(patient_str)

        if patient_id not in patients.keys():
            patients[patient_id] = []

        patients[patient_id].append(ct)


# %%
