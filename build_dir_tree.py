from utils import structure
from pathlib import Path
import argparse
import json
import pandas as pd

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Build a directory tree for the SUPERB dataset')

    parser.add_argument('--source', type=str)
    parser.add_argument('--target', type=str)
    parser.add_argument('--labels', type=str)
    parser.add_argument('--docs', type=str)


    args    = parser.parse_args()

    source  = Path(args.source)
    target  = Path(args.target)
    labels  = Path(args.labels)
    docs    = Path(args.docs)

    # Build the directory tree
    patient_has_vfa, not_in_excel, errors = structure.build_patient_directory_tree(source, docs, labels, target)

    # Save results to file
    with open("patient_has_vfa.json", "w") as f:
        json.dump(patient_has_vfa, f, indent=4)

    with open("not_in_excel.json", "w") as f:
        json.dump(not_in_excel, f, indent=4)
    
    with open("errors.csv", "w") as f:
        df = pd.DataFrame(errors)
        df.to_csv(f, index=False)
            
