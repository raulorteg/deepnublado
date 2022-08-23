# DeepNublado

A project with the aim of building an emulator for [Cloudy](https://nublado.org) for stand-alone use as well as use in large scale simulations. The necessary ground thruth for the machine learning experiments have been generated with [pyNublado](https://github.com/raulorteg/pyNublado).


## Setup

### Software requirements

We assume your system is equipped with the following dependencies:

* Python 3.8 or newer

#### System packages
tba.

#### Python modules
Furthermore, the following Python packages are needed:

* pytorch
* numpy (1.20.0 or newer)
* matplotlib
* tqdm

##### pip
The Python dependencies can be installed with `pip` like so:
```bash
pip3 install -r requirements.txt
```

##### conda
In Anaconda (or Miniconda) environments the requirements can be installed like so:
```bash
conda config --add channels conda-forge
conda install --yes --file requirements_conda.txt
```


