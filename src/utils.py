"""
This file contains various utility functions
"""

import numpy as np

from settings import \
    SCALER_MAX_INPUTS, \
    SCALER_MIN_INPUTS, \
    DEEPNUBLADO_INPUTS, \
    SETTING_TRANSFORM_LINE_DATA_OFFSET


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
