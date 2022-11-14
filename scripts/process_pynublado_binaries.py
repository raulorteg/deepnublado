import sys
import time
from pathlib import Path

import pandas as pd


def process_pynublado_binaries(entry_point: str):

    # manage exceptions
    if not Path(entry_point).exists():
        raise Exception(f"Given path entry point {entry_point} does not exist.")

    elif not Path(entry_point).is_dir():
        raise Exception(f"Given path entry point {entry_point} its not a directory.")

    print("-" * 20)
    print(" GENERATE THE DATASETS FROM PYNUBLADO BINARY OUTPUTS ")
    print(" pynublado documentation: https://raulorteg.github.io/pyNublado/")
    print(" pynublado repository: https://github.com/raulorteg/pyNublado")
    print(" deepnublado repository: https://github.com/raulorteg/deepnublado")
    print("-" * 20)

    # get sample paths inside given directory
    print("1. Scanning files")
    time_start_sample_scan = time.time()
    entry_dir = Path(entry_point)
    sample_paths = []
    for item in entry_dir.iterdir():
        if item.is_dir():
            sample_paths.append(item)

    time_end_sample_scan = time.time()
    time_delta = time_end_sample_scan - time_start_sample_scan
    print(
        f"\t Found {len(sample_paths)} samples in given directory ({time_delta:.3f} s):"
    )
    for sample_path in sample_paths:
        print(f"\t\t {sample_path.stem}")

    # perform join operation on inputs+status, stack the dataframes from different samples
    time_start_join_inputs_status = time.time()
    print("2. Generating the datasets")
    print("\t Classifier dataset (inputs+status)")
    print("\t\t Joining inputs+status")
    list_dataframes = []
    for sample_path in sample_paths:
        print(f"\t\t\t {sample_path}")
        inputs = pd.read_pickle(Path(sample_path, "inputs.pkl"), compression="infer")
        status = pd.read_pickle(Path(sample_path, "status.pkl"), compression="infer")

        data = inputs.set_index("id").join(status.set_index("id"))
        list_dataframes.append(data)

        del inputs, status, data

    time_end_join_inputs_status = time.time()
    time_delta = time_end_join_inputs_status - time_start_join_inputs_status
    print(f"\t\t Stacking samples ({time_delta:.3f} s)")

    # stacking
    time_start_stacking_classifier = time.time()
    classifier_data = pd.concat(list_dataframes)
    time_end_stacking_classifier = time.time()
    del list_dataframes
    time_delta = time_end_stacking_classifier - time_start_stacking_classifier
    print(f"\t\t Saving dataset ({time_delta:.3f} s)")

    # saving
    classifier_data.to_csv(Path(entry_point, "classifier.csv"), sep=",")
    mem_size_classifier = sys.getsizeof(classifier_data)
    del classifier_data
    time_end_saving_classifier = time.time()
    time_delta = time_end_saving_classifier - time_end_stacking_classifier
    print(f"\t\t Done ({time_delta:.3f} s)")

    # perform join operation on inputs+outputs, stack the dataframes from different samples
    time_start_join_inputs_outputs = time.time()
    print("\t Regressor dataset (inputs+outputs)")
    print("\t\t Joining inputs+status")
    list_dataframes = []
    for sample_path in sample_paths:
        print(f"\t\t\t {sample_path}")
        inputs = pd.read_pickle(Path(sample_path, "inputs.pkl"), compression="infer")
        status = pd.read_pickle(Path(sample_path, "status.pkl"), compression="infer")
        emis = pd.read_pickle(Path(sample_path, "emis.pkl"), compression="infer")
        emis.drop(columns=["index"], inplace=True)

        # drop outputs for whose status code was not EXITED_OK (0)
        data = inputs.set_index("id").join(status.set_index("id"))
        data = data.join(emis.set_index("id"))
        data = data[data.status == 0]
        data.drop(columns=["index", "status"], inplace=True)
        list_dataframes.append(data)

        del inputs, emis, data

    time_end_join_inputs_outputs = time.time()
    time_delta = time_end_join_inputs_outputs - time_start_join_inputs_outputs
    print(f"\t\t Stacking samples ({time_delta:.3f} s)")

    # stacking
    time_start_stacking_regressor = time.time()
    regressor_data = pd.concat(list_dataframes)
    time_end_stacking_regressor = time.time()
    del list_dataframes
    time_delta = time_end_stacking_regressor - time_start_stacking_regressor
    print(f"\t\t Saving dataset ({time_delta:.3f} s)")

    # saving
    regressor_data.to_csv(Path(entry_point, "regressor.csv"), sep=",")
    mem_size_regressor = sys.getsizeof(regressor_data)
    del regressor_data
    time_end_saving_regressor = time.time()
    time_delta = time_end_saving_regressor - time_end_stacking_regressor
    print(f"\t\t Done ({time_delta:.3f} s)")

    time_delta = time.time() - time_start_sample_scan
    print("-" * 20)
    print(f"Number of samples found: {len(sample_paths)}")
    print(f"Total elapsed time: {time_delta:.3f} s")
    print(f"Memory size (Classifier dset): {mem_size_classifier*1e-6} MBytes")
    print(f"Memory size (Regressor dset): {mem_size_regressor*1e-6} MBytes")


if __name__ == "__main__":

    # e.g python process_pynublado_binaries.py --path="./samples"
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        required=True,
        type=str,
        help="String path to directory where pynublado samples are stored",
    )
    args = parser.parse_args()
    process_pynublado_binaries(entry_point=args.path)
