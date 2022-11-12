import sys
import time

import numpy as np

sys.path.append("..")
from pathlib import Path

from src.settings import SETTING_TRAIN_FRACTION, SETTING_VAL_FRACTION


def deepnublado_dataset_split(
    entry_point: str, train_fraction: float = None, val_fraction: float = None
):

    start_time = time.time()
    filename = Path(entry_point)
    stem_filename = str(filename.stem)

    # manage exceptions
    if not filename.exists():
        raise Exception(f"Given path entry point {entry_point} does not exist.")

    elif not filename.is_file():
        raise Exception(f"Given path entry point {entry_point} its not a file.")

    elif ".csv" not in entry_point:
        raise Exception(
            f"Given path entry point {entry_point} its not a comma separated value (.csv) file."
        )
    # use the default partition fractions if not specified
    if not train_fraction:
        train_fraction = SETTING_TRAIN_FRACTION

    if not val_fraction:
        val_fraction = SETTING_VAL_FRACTION

    if train_fraction + val_fraction >= 1.0:
        total_fraction = train_fraction + val_fraction
        exception_msg = f"Train ({train_fraction:.2f} + validation ({val_fraction:.2f}) \
                fractions sum more than or the whole dataset ({train_fraction:.2f}+\
                {val_fraction:.2f}={total_fraction:.2f}>=1.0). Not enough data was left for the test set."
        raise Exception(exception_msg)

    test_fraction = 1 - (train_fraction + val_fraction)
    print("-" * 20)
    print(
        f"SPLIT TRAIN/VAL/TEST ({train_fraction:.3f}/{val_fraction:.3f}/{test_fraction:.3f}) PARTITIONS FROM PYNUBLADO .csv DATASET"
    )
    print(" pynublado documentation: https://raulorteg.github.io/pyNublado/")
    print(" pynublado repository: https://github.com/raulorteg/pyNublado")
    print(" deepnublado repository: https://github.com/raulorteg/deepnublado")
    print("-" * 20)

    train_filename = entry_point.replace(stem_filename, stem_filename + "_train")
    val_filename = entry_point.replace(stem_filename, stem_filename + "_val")
    test_filename = entry_point.replace(stem_filename, stem_filename + "_test")

    train_f = open(train_filename, "w+")
    val_f = open(val_filename, "w+")
    test_f = open(test_filename, "w+")
    partition_fobjects = [train_f, val_f, test_f]

    with open(filename, "r") as f:
        for i, line in enumerate(f):
            # write header in train, val and test file
            if i == 0:
                print(line, file=train_f, end="")
                print(line, file=val_f, end="")
                print(line, file=test_f, end="")
            else:
                idx = np.random.choice(
                    [0, 1, 2], p=[train_fraction, val_fraction, test_fraction]
                )
                print(line, file=partition_fobjects[idx], end="")

    train_f.close(), val_f.close(), test_f.close()

    end_time = time.time()
    delta_t = end_time - start_time
    print(f"Finished ({delta_t:.3f} s)")


if __name__ == "__main__":

    # e.g python train_val_test_split.py --filepath="./samples/classifier.csv"
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
    parser.add_argument(
        "--val_fraction",
        required=False,
        type=float,
        help="Float value in (0,1) representing the ratio of the total data to be used in validation",
    )
    args = parser.parse_args()
    deepnublado_dataset_split(
        entry_point=args.path,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )
