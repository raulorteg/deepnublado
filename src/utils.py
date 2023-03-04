"""
This file contains various utility functions
"""

import os
import pickle
import logging
import torch
import numpy as np
import pandas as pd

from settings import \
    SCALER_MAX_INPUTS, \
    SCALER_MIN_INPUTS, \
    DEEPNUBLADO_INPUTS, \
    DEEPNUBLADO_REGRESSOR_OUTPUTS, \
    SETTING_TRANSFORM_LINE_DATA_OFFSET, \
    SETTING_DATA_PRODUCTS_SUBDIR


# -----------------------------------------------------------------
# functions to scale and re-scale Cloudy inputs /parameters
# -----------------------------------------------------------------
def utils_scale_inputs(parameters):
    """
    Scales all parameters to [0,1], Also filters the SCALER_MIN/MAX_INPUTS by DEEPNUBLADO_INPUTS
    :param parameters: 2D array of all inputs / parameters
    :return: re-scaled 2D inputs array
    """

    for i, input_name in enumerate(DEEPNUBLADO_INPUTS):

        a = SCALER_MIN_INPUTS[input_name]
        b = SCALER_MAX_INPUTS[input_name]

        parameters[:, i] = (parameters[:, i] - a) / (b - a)

    return parameters


def utils_rescale_inputs(parameters):
    """
    Re-scales all parameters to original limits.
    :param parameters: 2D array of all inputs / parameters
    :return: re-scaled 2D inputs array
    """

    for i, input_name in enumerate(DEEPNUBLADO_INPUTS):

        a = SCALER_MIN_INPUTS[input_name]
        b = SCALER_MAX_INPUTS[input_name]

        parameters[:, i] = parameters[:, i] * (b - a) + a

    return parameters


def utils_rescale_inputs_single(p):
    """
    Re-scales a single given input / parameter vector
    :param p: 1D array of all inputs / parameters
    :return: re-scaled 1D inputs array
    """

    for i, input_name in enumerate(DEEPNUBLADO_INPUTS):

        a = SCALER_MIN_INPUTS[input_name]
        b = SCALER_MAX_INPUTS[input_name]

        p[i] = p[i] * (b - a) + a

    return p


# -----------------------------------------------------------------
# transform and re-transform Cloudy emission line outputs
# -----------------------------------------------------------------
def utils_transform_line_data(emission_lines):
    """
    A simple first approach to transform the emission line data
    in order to reduce the high dynamic range from 30 orders of
    magnitude 2: we add an offset and take the log.

    TODO: test other methods? E.g. Raul's min-max scaling?

    :param emission_lines: 2D numpy array containing all line data
    :return: transformed 2D numpy array
    """

    return np.log10(emission_lines + SETTING_TRANSFORM_LINE_DATA_OFFSET)


def utils_de_transform_line_data(emission_lines):
    """
    Reverse of the line data transformation function.
    :param emission_lines:  2D numpy array containing all line data
    :return: de-transformed 2D numpy array
    """

    return 10**emission_lines - SETTING_TRANSFORM_LINE_DATA_OFFSET


# -----------------------------------------------------------------
# save test set inputs and lines (true & inference)
# -----------------------------------------------------------------
def utils_save_test_data(parameters, lines_true, lines_gen,
                         path: str,
                         epoch: int,
                         prefix: str = 'test'):
    """
    Writes the inferred test set data, the corresponding ground truth,
    and input parameters to disk. This is currently done in two ways:
    1) as numpy arrays into npy files
    2) as pandas data frames into pickle files

    :param parameters: 2D numpy array containing the input parameters
    :param lines_true: 2D numpy array containing simulated emission line data
    :param lines_gen: 2D numpy array containing inferred emission line data
    :param path:  output dir
    :param epoch: epoch
    :param prefix: 'test' / 'best prefix'
    :return: -
    """

    # 1. save data as numpy arrays
    parameter_filename = F"{prefix}_inputs_{epoch}_epochs.npy"
    lines_true_filename = F"{prefix}_lines_true_{epoch}_epochs.npy"
    lines_gen_filename = F"{prefix}_lines_gen_{epoch}_epochs.npy"

    parameter_path = os.path.join(path, parameter_filename)
    lines_true_path = os.path.join(path, lines_true_filename)
    lines_gen_path = os.path.join(path, lines_gen_filename)

    print("\nSaving results in the following npy files:\n")
    print(F"  {parameter_path}")
    print(F"  {lines_true_path}")
    print(F"  {lines_gen_path}")

    np.save(parameter_path, parameters)
    np.save(lines_true_path, lines_true)
    np.save(lines_gen_path, lines_gen)

    # 2. save as pandas data frames (containing inputs, lines, and their headers)
    stack_true = np.hstack((parameters, lines_true))
    stack_gen = np.hstack((parameters, lines_gen))

    column_names = DEEPNUBLADO_INPUTS + DEEPNUBLADO_REGRESSOR_OUTPUTS
    lines_true_df = pd.DataFrame(stack_true, columns=column_names)
    lines_gen_df = pd.DataFrame(stack_gen, columns=column_names)

    lines_true_filename = F"{prefix}_lines_true_{epoch}_epochs.pkl"
    lines_gen_filename = F"{prefix}_lines_gen_{epoch}_epochs.pkl"

    lines_true_path = os.path.join(path, lines_true_filename)
    lines_gen_path = os.path.join(path, lines_gen_filename)

    lines_true_df.to_pickle(lines_true_path)
    lines_gen_df.to_pickle(lines_gen_path)

    print("\nSaving results as data frames in the following pickle files:\n")
    print(F"  {lines_true_path}")
    print(F"  {lines_gen_path}\n")


# -----------------------------------------------------------------
# save model state
# -----------------------------------------------------------------
def utils_save_model(state,
                     path: str,
                     n_epoch: int,
                     best_model: bool = False,
                     file_name: str = None):
    """
    Saves the model weights for later use.

    :param state: state, i.e. the weights, of the model in use
    :param path: output dir
    :param n_epoch: number of training epochs
    :param best_model: bool
    :param file_name: string of model file name
    :return: nothing
    """

    # if no file name is provided, construct one here
    if file_name is None:
        file_name = F"model_{n_epoch}_epochs.pth.tar"

        if best_model:
            file_name = 'best_' + file_name

    path = os.path.join(path, file_name)
    torch.save(state, path)
    print(F"\nSaved model to:\n  {path}")


# -----------------------------------------------------------------
# save loss function as numpy object
# -----------------------------------------------------------------
def utils_save_loss(loss_array, path: str, n_epoch: int, prefix: str = 'train'):
    """
    Saves a given loss array as a npy file.

    :param loss_array: 1D numpy array
    :param path: path to the run directory (/data_products)
    :param n_epoch: total number of epochs
    :param prefix: train / val prefix
    :return: nothing
    """

    file_name = prefix + F"_loss_{n_epoch}_epochs.npy"
    path = os.path.join(path, file_name)
    np.save(path, loss_array)
    print(F"\nSaved {prefix} loss function to:\n  {path}")


# -----------------------------------------------------------------
# Write argparse config to ascii file
# -----------------------------------------------------------------
def utils_save_config_to_log(config):

    c = "\n  Contents of the config object\n\n"

    for arg in vars(config):
        line = F"  {str(arg):18}  {getattr(config, arg)} \n"
        c += line

    logging.info(c)


# -----------------------------------------------------------------
# Write argparse config to binary file (to re-use later)
# -----------------------------------------------------------------
def utils_save_config_to_file(config, file_name='config.dict'):
    """
    Saves the config object locally for later use

    :param config: argparse config object
    :param file_name:
    :return: nothing
    """

    p = os.path.join(config.run_dir, SETTING_DATA_PRODUCTS_SUBDIR, file_name)

    print('\nWriting config to binary file:')
    print('  ' + p)

    with open(p, 'wb') as f:
        pickle.dump(config, f)


# -----------------------------------------------------------------
# Load argparse config from binary file
# -----------------------------------------------------------------
def utils_load_config(path, file_name='config.dict'):
    """
    Loads a saved argparse config object from disk to resume training or
    to run inference on fully trained models

    :param path: path to data products directory or to file
    :param file_name: config file name
    :return: config object
    """

    if path.endswith(file_name):
        p = path
    else:
        p = os.path.join(path, file_name)

    print('\nLoading config object from file:\n')
    print('  ' + p)

    with open(p, 'rb') as f:
        config = pickle.load(f)

    return config
