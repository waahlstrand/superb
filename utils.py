import os
from glob import glob
from tqdm import tqdm
from lxml import etree
from typing import *
import pydicom
from preprocessing.xml_to_dict import XmlDictConfig
import shutil
from pathlib import Path
from PIL import Image

ANNOTATION_TAG = (0x0021, 0x1001)
PATIENT_TAG = (0x0010, 0x0020)
PIXEL_SPACING_TAG = (0x0028, 0x0030)


def image_id_to_excel_id(id: str) -> str:

    if "MO" in id:
        return "MO"+id.replace("MO", "").zfill(4)
    elif "M0" in id:
        return "MO"+id.replace("M0", "").zfill(4)
    elif "mo" in id:
        return "MO"+id.replace("mo", "").zfill(4)
    else:
        return id

def patient_annotations(root: str):

    files = glob(os.path.join(root, "*.dcm"))
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

    # Read file
    ds = pydicom.dcmread(file)

    # Extract xml string
    xml_str = ds[ANNOTATION_TAG].value
    
    # Parse the xml string and return a dictionary
    annotation = parse_annotation_xml_str(xml_str)

    return annotation

def parse_annotation_xml_str(xml_str: str) -> Dict[str, Any]:

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


def build_patient_directory_tree(src: str, tgt: str):

    files = glob(src, "*.dcm")
    groups: Dict[str, List[str]] = {}
    for file in tqdm(files):
        ds = pydicom.dcmread(file)

        if not ds.PatientID in groups.keys():
            groups[ds.PatientID] = []
        
        groups[ds.PatientID].append(file)

    for patient_id, group in tqdm(groups.items()):
        
        # Create the directory tree
        # os.makedirs(os.path.join(base_dir, f"{patient_id}", "dicom"))

        # Copy the files to the new location
        for file in group:
            filename = file.split("/")[-1]

            ds = pydicom.dcmread(file)

            if ANNOTATION_TAG in ds:
                new_directory = os.path.join(tgt, f"{patient_id}", "dicom", "vfa")
                new_file = os.path.join(new_directory, filename)
            else:
                new_directory = os.path.join(tgt, f"{patient_id}", "dicom", "reports")
                new_file = os.path.join(new_directory, filename)

            if not os.path.exists(new_directory):
                os.makedirs(new_directory)
            
            if not os.path.exists(new_file):
                shutil.copyfile(file, new_file)

def write_annotation_file(filename: Path, x: List[float], y: List[float]):
    pass

def extract_and_build_vfa_images(patient_root: Path):

    files = patient_root.glob("./**/dicom/vfa/*.dcm")

    for file in tqdm(files):

        # Create a new folder
        new_directory = file.parent / "image"

        if not os.path.exists(new_directory):
            os.makedirs(new_directory)
        
        # Extract image from DICOM
        ds = pydicom.dcmread(file)
        id = ds[PATIENT_TAG].value
        x, y = ds[PIXEL_SPACING_TAG].value

        # Save new image
        new_file = new_directory / Path(id + ".tiff")
        Image.fromarray(ds.pixel_array).save(new_file)





