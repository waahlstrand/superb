#%%
import os
from glob import glob
from tqdm import tqdm
from lxml import etree
from typing import *
import pydicom
from preprocessing.xml_to_dict import XmlDictConfig
import preprocessing.labels as labelling
import shutil
from pathlib import Path
from PIL import Image
import pandas as pd
import json 

ANNOTATION_TAG = (0x0021, 0x1001)
PATIENT_TAG = (0x0010, 0x0020)
PIXEL_SPACING_TAG = (0x0028, 0x0030)


def patient_annotations(root: Path):
    """
    Read the annotations from all DICOM files in a directory and return a list of dictionaries.
    
    Args:
        root (Path): The path to the directory containing the DICOM files.
        
    Returns:
        annotations (list): A list of dictionaries containing the annotations.
    """

    files = root.glob("*.dcm")
    pads = []

    for file in tqdm(files):

        ds = pydicom.dcmread(file)

        if ANNOTATION_TAG in ds:

            pad = patient_annotation(ds)

            pads.append(pad)

    return pads


def patient_annotation(ds: pydicom.Dataset):

    pad = parse_annotation_xml_str(ds[ANNOTATION_TAG].value)

    return pad


def get_annotation(file: str) -> Dict[str, Any]:
    """
    Read the annotation from a DICOM file and return a dictionary.
    
    Args:
        file (str): The path to the DICOM file.
    
    Returns:
        annotation (dict): A dictionary containing the annotation.
    """

    # Read file
    ds = pydicom.dcmread(file)

    # Extract xml string
    xml_str = ds[ANNOTATION_TAG].value
    
    # Parse the xml string and return a dictionary
    annotation = parse_annotation_xml_str(xml_str)

    return annotation

def parse_annotation_xml_str(xml_str: str) -> Dict[str, Any]:
    """
    Parse the annotation XML string and return a corresponding dictionary.
    
    Args:
        xml_str (str): The annotation XML string.
        
    Returns:
        annotation (dict): A dictionary containing the annotation.
    """

    root = etree.fromstring(xml_str)

    d = XmlDictConfig(root)

    annotation = {}

    for k in d.keys():

        if k.startswith("T") or k.startswith("L"):

            vertebra = {}
            points = {}
            
            qm = d[k]["QM"]
            points["x"] = [float(qm[i]) for i in qm.keys() if i.startswith("X")]
            points["y"] = [float(qm[i]) for i in qm.keys() if i.startswith("Y")]

            vertebra.update({
                **d[k],
                **points
            })

            annotation.update({k: vertebra})

    return annotation


def build_patient_directory_tree(dcm_root: Path, labels_path: Path, target_root: Path):
    """
    Build a directory tree for the patient data.
    
    Structure:
        - patient_id
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

    is_vfa_dicom = lambda x: ANNOTATION_TAG in x

    # Group files by patient
    groups: Dict[str, List[Path]] = {} # Patient ID -> List of files

    for file in tqdm(files, desc="Grouping files by patient"):
        ds = pydicom.dcmread(file)

        if not ds.PatientID in groups.keys():
            groups[ds.PatientID] = []
        
        groups[ds.PatientID].append(file)

    progress = tqdm(groups.items(), desc="Building directory tree")
    for image_patient_id, group in progress:

        progress.set_description(f"Building directory tree for {image_patient_id}")

        # Convert the patient ID used for images to the patient ID used in the labels
        patient_id = labelling.image_id_to_excel_id(image_patient_id)

        lateral_file_dir =  target_root / patient_id / "lateral" 
        report_file_dir  = target_root / patient_id / "reports"

        if not os.path.exists(lateral_file_dir):
            os.makedirs(lateral_file_dir)

        if not os.path.exists(report_file_dir):
            os.makedirs(report_file_dir)

        # Copy the files to the new location
        for file in group:

            d = pydicom.dcmread(file)

            # If the file is a VFA dicom file
            #  - Create a renamed dicom file
            #  - Download the image as an easy access tiff
            #  - Get the label for the image
            #  - Save all the files
            if is_vfa_dicom(d):

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
                Image.fromarray(d.pixel_array).save(new_image_file)

                ## Handle the label information sources
                dx, dy     = d.PixelSpacing
                annotation = patient_annotation(d)
                label      = labelling.excel_to_records(labels[labels["ID"] == patient_id])

                # Make sure there is a label for the patient
                if len(label) == 1:
                    
                    label = label[0]
                    for vertebra in labelling.VERTEBRA_NAMES:
                        if vertebra in label and vertebra in annotation:
                            label[vertebra].update(annotation[vertebra])
                        elif vertebra not in label and vertebra in annotation:
                            label[vertebra] = annotation[vertebra]
                        else:
                            label[vertebra] = {}

                    label.update({"pixel_spacing": [dx, dy]})

                    with open(new_label_file, "w") as f:
                        json.dump(label, f, indent=4)
                
                else:
                    print(f"Could not find label for patient {patient_id}")

            # If the file is not a VFA dicom file, it is a report
            else:
                # Create a renamed dicom file
                new_dcm_file = report_file_dir / file.name

                # Save the dicom file
                d.save_as(new_dcm_file)

            # Release memory of dicom file
            del d