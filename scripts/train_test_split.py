import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append("..")
from pathlib import Path

from src.settings import SETTING_TRAIN_FRACTION


def deepnublado_dataset_split(entry_point: str, train_fraction: float = None):

    start_time = time.time()
    filename = Path(entry_point)
    stem_filename = str(filename.stem)

    # manage exceptions
    if not filename.exists():
        raise Exception(f"Given path entry point {entry_point} does not exist.")

    elif not filename.is_file():
        raise Exception(f"Given path entry point {entry_point} its not a file.")

    elif not ".csv" in entry_point:
        raise Exception(
            f"Given path entry point {entry_point} its not a comma separated value (.csv) file."
        )

    if not train_fraction:
        train_fraction = SETTING_TRAIN_FRACTION

    if (train_fraction <= 0.0) or (train_fraction >= 1.0):
        raise Exception(
            f"Train fraction out of range, only values in (0.0,1.0) are valid. Value given: {train_fraction}"
        )

    test_fraction = 1 - train_fraction
    print("-" * 20)
    print(
        f" SPLIT TRAIN/TEST ({train_fraction:.3f}/{test_fraction:.3f}) PARTITIONS FROM PYNUBLADO .csv DATASET"
    )
    print(" pynublado documentation: https://raulorteg.github.io/pyNublado/")
    print(" pynublado repository: https://github.com/raulorteg/pyNublado")
    print(" deepnublado repository: https://github.com/raulorteg/deepnublado")
    print("-" * 20)

    train_filename, test_filename = entry_point.replace(
        stem_filename, stem_filename + "_train"
    ), entry_point.replace(stem_filename, stem_filename + "_test")

    train_f = open(train_filename, "w+")
    test_f = open(test_filename, "w+")
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            # write header in both train and test file
            if i == 0:
                print(line, file=train_f, end="")
                print(line, file=test_f, end="")
            else:
                if train_fraction >= np.random.uniform():
                    print(line, file=train_f, end="")
                else:
                    print(line, file=test_f, end="")

    train_f.close(), test_f.close()

    end_time = time.time()
    delta_t = end_time - start_time
    print(f"Finished ({delta_t:.3f} s)")


if __name__ == "__main__":

    # e.g python train_test_split.py --filepath="./samples/classifier.csv"
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        required=True,
        type=str,
        help="String path to directory where pynublado dataset is stored",
    )
    parser.add_argument(
        "--train_fraction",
        required=False,
        type=float,
        default=None,
        help="Float value in (0,1) representing the ratio of the total data to be used in training.",
    )
    args = parser.parse_args()
    deepnublado_dataset_split(entry_point=args.path, train_fraction=args.train_fraction)
