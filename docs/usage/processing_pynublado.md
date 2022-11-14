
## Processing Pynublado

- *pynublado repository*: [](https://github.com/raulorteg/pyNublado)
- *pynublado documentation*: [](https://raulorteg.github.io/pyNublado/)

The following instructions assume the following file structure at $pathtosamples , which is the standard file strcuture outputted by pynublado.

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

The output of the processing script is two .csv files containing the data needed for the classifier and regressor (classifier.csv, regressor.csv). These two files are saved at pathtosamples/classifier.csv and pathtosamples/regressor.csv. Resulting in the following file structure:

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

```bash
python train_test_split.py --path=<path_to_samples>/<dataset>.csv --train_fraction=<train_fraction> --val_fraction=<val_fraction>
```

Options:

* `path`: [str] Path to the FILENAME with processed samples (classifier.csv or regressor.csv) to be splitted into the train/test partitions.
* `train_fraction`: [float, Optional] Fraction of the total dataset to be used as the training set (e.g 0.8 -> 80%), the rest will be used for testing and validation. The default value is defined in `src/settings.py` as 0.8.
* `val_fraction`: [float, Optional] Fraction of the total dataset to be used as the validation set (e.g 0.1 -> 10%), the sum of the train and val fractions must be less than the total, since the remainer will be used for testing. (e.g 1-0.8-0.1=0.1 -> 10% test percentage). The default value is defined in `src/settings.py` as 0.1.

The output of the splitting script is three .csv files containing the two partitions of the file given using the ratio defined by the train\_fraction and val\_fraction parameters. These files are saved at pathtosamples/dataset\_train.csv, pathtosamples/dataset\_val.csv and pathtosamples/dataset\_test.csv. Resulting in the following file structure:

```bash
.
├── regressor_train.csv      <---
├── regressor_test.csv       <---
├── regressor_val.csv        <---
├── regressor.csv      
├── classifier_train.csv     <---
├── classifier_test.csv      <---
├── classifier_val.csv       <---
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
