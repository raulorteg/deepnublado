
## Processing datasets from pynublado:
------------------------------------------
- *pynublado repository*: https://github.com/raulorteg/pyNublado
- *pynublado documentation*: https://raulorteg.github.io/pyNublado/

The following instructions assume the following file structure at _<path_to_samples>_ , which is the standard file strcuture outputted by pynublado.

```bash
.
├── sample_N10005
│   ├── cont.pkl
│   ├── emis.pkl
│   ├── inputs.pkl
│   ├── parameters_N10005.npy
│   └── status.pkl
├── sample_N11005
│   └── ...
├── sample_N12005
│   └── ...
...
└── sample_N5005
    └── ...
```

1. Create the datasets (classifier & regressor)
```bash
python process_pynublado_binaries.py --path=<path_to_samples>
```

Options:

* `path`: [str] Path to the folder containg the samples from _pynublado_, note that the script expects the file structure shown above.

The output of the processing script is two .csv files containing the data needed for the classifier and regressor (_classifier.csv_, _regressor.csv_). These two files are saved at _<path_to_samples>/classifier.csv_ and _<path_to_samples>/regressor.csv_. Resulting in the following file structure:

```bash
.
├── regressor.csv      <---
├── classifier.csv     <---
├── sample_N10005
│   └── ...
├── sample_N11005
│   └── ...
├── sample_N12005
│   └── ...
...
└── sample_N5005
    └── ...
```

2. Splitting the dataset in Training/Testing sets.
```
python train_test_split.py --path=<path_to_samples>/<dataset>.csv --train_fraction=<train_fraction>
```

Options:

* `path`: [str] Path to the FILENAME with processed samples (_classifier.csv_ or _regressor.csv_) to be splitted into the train/test partitions.
* `train_fraction`: [float, Optional] Fraction of the total dataset to be used as the training set (e.g 0.8 -> 80%), the rest will be used for testing (1-0.8=0.2 -> 20%). The default value is defined in ```src/settings.py``` as 0.8.

The output of the splitting script is two .csv files containing the two partitions of the file given using the ratio defined by the <train_fraction> parameter. These two files are saved at _<path_to_samples>/<dataset>_train.csv_ and _<path_to_samples>/<dataset>_test.csv_. Resulting in the following file structure:

```bash
.
├── regressor_train.csv      <---
├── regressor_test.csv       <---
├── regressor.csv      
├── classifier_train.csv     <---
├── classifier_test.csv      <---
├── classifier.csv     
├── sample_N10005
│   └── ...
├── sample_N11005
│   └── ...
├── sample_N12005
│   └── ...
...
└── sample_N5005
    └── ...
```





