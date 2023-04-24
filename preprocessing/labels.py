import pandas as pd
from typing import *
import json
from pathlib import Path

THORACIC = [
    "T12", "T11", "T10", "T9", "T8", "T7", "T6", "T5", "T4"
]

LUMBAR = [
    "L4", "L3", "L2", "L1"
]

VERTEBRA_NAMES = [
    *LUMBAR,
    *THORACIC
]

COLS = [
    "GRAD_MORF",
    "GRAD_VISUELL",
    "TYP",
    "_EJ_BEDÖMBAR",
    "KOMMENTAR"
]

OTHER_COLS = [
    "ULL",
    "K",
    "R",
    "A",
    "DF",
    "S",
    "BAC",
    "FR",
    "SK",
    "VLS",
    "HLS",
    "VBS",
    "HBS",
    "GRANSKAD",
    "Reader",
    "EJ_BEDÖMBAR",
]

def excel_to_dict(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """
    Converts a pandas dataframe to a dictionary.
    """

    df = df.iloc[1:] # Remove header
    df = df.fillna(0) # Replace NaN with 0

    df = df.set_index("ID", inplace=False) # Set ID as index

    # Normalize the column names
    df.columns = df.columns.str.strip()

    # Create a dictionary
    d = {}
    for idx, row in df.iterrows():
        d[idx] = {}

        for vertebra in VERTEBRA_NAMES:

            d[idx][vertebra] = {}
            
            for col in COLS:

                d[idx][vertebra][col] = 0.0
                
                if hasattr(row, vertebra + col):
                    d[idx][vertebra][col] = getattr(row, vertebra + col)
                else:
                    print(row.to_string())
                    raise ValueError(f"Could not find {vertebra + col} in row {row}")

        for col in OTHER_COLS:
            d[idx][col] = getattr(row, col)

    return d



def excel_to_records(df: pd.DataFrame, only_auditable=False) -> List[Dict[str, str]]:
    """
    Converts a pandas dataframe to a list of dictionaries.
    """
    # df = df.iloc[1:]
    if only_auditable:
        df = df[~df["EJ_BEDÖMBAR"].notna()]

    df = df.fillna(0)
    df.columns = df.columns.str.strip()

    records = []
    for idx, row in df.iterrows():
        record = {}
        record["id"] = row["ID"]

        for vertebra in VERTEBRA_NAMES:

            record[vertebra] = {}
            
            for col in COLS:
                
                if col == "_EJ_BEDÖMBAR":
                    record[vertebra][col] = None
                else:
                    record[vertebra][col] = getattr(row, vertebra + col)

        for col in OTHER_COLS:
            record[col] = getattr(row, col)

        records.append(record)

    return records

def excel_to_vertebra_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    """
    Converts a pandas dataframe to a dataframe with one row per vertebra.
    """
    records = []
    for idx, row in df.iterrows():
        for vertebra in VERTEBRA_NAMES:

            record = {}
            record["id"] = row["ID"]
            record["vertebra"] = vertebra
            
            for col in COLS:
                
                if (vertebra == "L4" or vertebra == "L3") and (col == "GRAD_VISUELL" or col == "_EJ_BEDÖMBAR"):
                    continue
                else:
                    record[col] = getattr(row, vertebra + col)

            for col in OTHER_COLS:
                record[col] = getattr(row, col)

            records.append(record)

    vf = pd.DataFrame(records)

    vf.rename(columns={
        "_EJ_BEDÖMBAR": "KOTA_EJ_BEDÖMBAR",
        "EJ_BEDÖMBAR": "PATIENT_EJ_BEDÖMBAR"
        }, inplace=True)
    
    return vf

def image_id_to_excel_id(id: str) -> str:
    """
    Converts an image id to an excel id. Covers the following cases:
    MO0001, M0001, mo0001, mo001, mo01, mo1, m01, m1, m0001, m001, m01, m1

    Args:
        id: Patient id.
    """

    if "MO" in id:
        return "MO"+id.replace("MO", "").zfill(4)
    elif "M0" in id:
        return "MO"+id.replace("M0", "").zfill(4)
    elif "mo" in id:
        return "MO"+id.replace("mo", "").zfill(4)
    else:
        return id
    
has_grad_morf = lambda x: x["GRAD_MORF"] != 0
has_vertebra_with_grad_morf = lambda x: any([x[vertebra]["GRAD_MORF"] != 0 for vertebra in VERTEBRA_NAMES])

def build_conversion_dicts(path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Builds two dictionaries, 
    one mapping image ids to excel ids and one mapping excel ids to image ids.

    Args:
        path: Path to the folder containing the patient directories.
    """

    image_to_excel = {}
    excel_to_image = {}
    for p in Path(path).glob("*"):
        
        image_to_excel[p.name] = image_id_to_excel_id(str(p.name))
        excel_to_image[image_id_to_excel_id(str(p.name))] = p.name


    return image_to_excel, excel_to_image
