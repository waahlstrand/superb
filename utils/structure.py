#%%
import os
from tqdm import tqdm
from typing import *
import numpy as np
import pydicom
import utils.labels as labelling
import shutil
from pathlib import Path
from PIL import Image
import pandas as pd
import json 
import itk
import cv2
from utils.extract import Extractor, Mode
from itertools import product

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


def get_patient_vfa_document(path: Path) -> Dict[str, Union[Path, None]]:

    pdfs = list(path.rglob("*.pdf"))
    word_docs = list(path.rglob("*.docx"))
    files = pdfs + word_docs

    patient_docs = {}
    for file in files:
        moid = labelling.doc_name_to_moid(file, suffix=file.suffix)
        patient_docs.update({moid: file})

    return patient_docs


def build_patient_directory_tree(
        dcm_root: Path, 
        vfa_document_root: Path,
        labels_path: Path, 
        target_root: Path,
        file_directory: Path = Path("./images")
        ):
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
    vfas    = get_patient_vfa_document(vfa_document_root)

    is_vfa_dicom = lambda x: x[PIXEL_SPACING_TAG].value is not None
    has_vfa_annotation = lambda x: x in vfas.keys()
    has_full_annotation = lambda vertebrae, names: len(vertebrae) == len(names)

    degrees = [3, 4, 5]
    thresholds = [10, 11, 12, 13, 14, 15]
    products = list(product(degrees, thresholds))

    # Group files by patient
    lacks_vfa = []
    errors = set()

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
            new_dcm_file    = lateral_file_dir / "patient.dcm"

            # Download the image as an easy access tiff
            new_image_file  = lateral_file_dir / "patient.tiff"

            # Get the label for the image
            new_label_file  = lateral_file_dir / "patient.json"

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
            if has_vfa_annotation(patient_id):

                document = vfas[patient_id]

                # Get new document file name
                new_document_file = report_file_dir / ("patient" + document.suffix)

                # Save the document
                shutil.copy(document, new_document_file)

                template = cv2.imread(str(new_image_file), cv2.IMREAD_GRAYSCALE)

                try:
                    extractor = Extractor(document, template, file_directory)
                except Exception as e:
                    errors.add((patient_id, file, e))
                    continue
                
                # Some heuristics to get the right degree and threshold for certain patients
                if int(patient_id.removeprefix("MO")) > 369 and extractor.mode == Mode.WORD:
                    size_factor = 0.921
                    degree = 5
                    threshold = 10
                else:
                    size_factor = 0.9
                    degree = 5
                    threshold = 10

                vertebrae = np.array([])

                # Run a grid search over the degrees and thresholds
                # to find the first combination that works
                names = None
                for degree, threshold in products:
                    progress.set_description(f"{patient_id}, threshold {threshold}, degree {degree}")

                    try:
                        vertebrae, names = extractor.vertebrae(degree=degree, residual_threshold=threshold, size_factor=size_factor)
                    except ValueError as e:
                        continue
                    except Exception as e:
                        errors.add((patient_id, file, e))
                        continue
                    else:
                        break

                if len(vertebrae) == 0:
                    errors.add((patient_id, file, "No vertebrae extracted."))

                
                # If we have names and coordinates, add both
                if names is not None and len(names) == len(vertebrae):
                    for name, vertebra in zip(names, vertebrae):
                        label[name].update({
                            "coordinates": vertebra.tolist(),
                        })

                # If we have all vertebrae, we already know the names
                elif len(vertebrae) == len(labelling.VERTEBRA_NAMES):
                    for name, vertebra in zip(labelling.VERTEBRA_NAMES[::-1], vertebrae):
                        label[name].update({
                            "coordinates": vertebra.tolist(),
                        })
                
                # If we have coordinates but no names, add only coordinates
                label.update({"keypoints": vertebrae.tolist()})

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

