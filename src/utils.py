"""
This file contains various utility functions
"""

import numpy as np

from settings import \
    SCALER_MAX_INPUTS, \
    SCALER_MIN_INPUTS, \
    DEEPNUBLADO_INPUTS


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
