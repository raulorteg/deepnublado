"""
This file contains methods used to set up a single training run
"""
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader

from data import DeepNubladoData
from utils import utils_scale_inputs, utils_transform_line_data
from settings import \
    SETTING_DATA_PRODUCTS_SUBDIR, \
    SETTING_PLOTS_SUBDIR, \
    SETTING_LINE_FILE, \
    DEEPNUBLADO_REGRESSOR_OUTPUTS, \
    DEEPNUBLADO_INPUTS, \
    SETTING_NORMALISE_INPUTS, \
    SETTING_TRANSFORM_LINE_DATA


def setup_main(config):
    """
    Main driver method to set up a run

    :param config: argparse object
    :return: 3 pytorch data loaders
    """

    # TODO: additional randomisation of data?
    # TODO: load continuum data & run outcomes?

    setup_run_directories(config)
    setup_log_file(config)
    inputs, lines = setup_load_data(config)

    if SETTING_NORMALISE_INPUTS:
        inputs = utils_scale_inputs(parameters=inputs)

    if SETTING_TRANSFORM_LINE_DATA:
        lines = utils_transform_line_data(emission_lines=lines)

    # additional randomisation of data here?

    # data loaders
    training_data = DeepNubladoData(parameters=inputs, emission_lines=lines, split='train')
    validation_data = DeepNubladoData(parameters=inputs, emission_lines=lines, split='val')
    testing_data = DeepNubladoData(parameters=inputs, emission_lines=lines, split='test')

    train_loader = DataLoader(training_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(validation_data, batch_size=config.batch_size)
    test_loader = DataLoader(testing_data, batch_size=config.batch_size)

    return train_loader, val_loader, test_loader


def setup_log_file(config):
    """
    Sets up and configures the logger package. Here, we create a log
    file in the run dir and define the log format string.
    :param config: config object
    :return: -
    """

    log_file = os.path.join(config.run_dir, 'run_log')
    logging.basicConfig(filename=log_file,
                        level=logging.INFO,
                        format='%(asctime)s - %(message)s')
    logging.info("Setting up the run")


def setup_load_data(config):
    """
    Imports data file as pandas data frame for easy manipulation, then
    converts and returns data as numpy arrays

    :param config: argparse object
    :return: nd arrays for parameters/inputs and emission lines
    """

    line_file_path = os.path.join(config.data_dir, SETTING_LINE_FILE)
    all_df = pd.read_csv(line_file_path)

    inputs_df = all_df.filter(DEEPNUBLADO_INPUTS)
    lines_df = all_df.filter(DEEPNUBLADO_REGRESSOR_OUTPUTS)

    inputs = inputs_df.to_numpy()
    lines = lines_df.to_numpy()

    return inputs, lines


def setup_run_directories(config):
    """
    Creates a unique output directory featuring a run_id

    :param config: argparse object
    :return: -
    """

    run_id = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    config.run_dir = os.path.join(config.out_dir, run_id)

    d = os.path.join(config.run_dir, SETTING_DATA_PRODUCTS_SUBDIR)
    p = os.path.join(config.run_dir, SETTING_PLOTS_SUBDIR)

    print("\nCreating directories:\n")
    print(F"  {config.run_dir}")
    print(F"  {d}")
    print(F"  {p}")

    os.makedirs(config.run_dir, exist_ok=False)
    os.makedirs(d, exist_ok=False)
    os.makedirs(p, exist_ok=False)





