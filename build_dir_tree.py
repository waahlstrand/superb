import utils
from pathlib import Path
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Build a directory tree for the SUPERB dataset')

    parser.add_argument('--source', type=str)
    parser.add_argument('--target', type=str)
    parser.add_argument('--labels', type=str)

    args = parser.parse_args()

    source = Path(args.source)
    target = Path(args.target)
    labels = Path(args.labels)

    # Build the directory tree
    utils.build_patient_directory_tree(source, target, labels)
