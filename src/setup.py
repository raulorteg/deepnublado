"""
This file contains methods used to set up a single training run
"""
import os
from datetime import datetime

from settings import \
    SETTING_MAIN_OUTPUT_DIR, \
    SETTING_DATA_PRODUCTS_SUBDIR, \
    SETTING_PLOTS_SUBDIR, \
    SCALER_MAX_INPUTS, \
    SCALER_MIN_INPUTS


def setup_main(config):

    setup_load_data(config)
    setup_run_directories(config)


def setup_load_data(config):

    # 1. locate and load data files

    # 2. possible transform parameter data
    #   see settings.SCALER_MIN_INPUTS
    #   and settings.SCALER_MAX_INPUTS

    # 3. possibly filter data
    #    see settings.DEEPNUBLADO_INPUTS
    #    and settings.DEEPNUBLADO_REGRESSOR_OUTPUTS

    pass


def setup_run_directories(config):

    run_id = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    config.run_dir = os.path.join(config.out_dir, run_id)

    d = os.path.join(config.run_dir, SETTING_DATA_PRODUCTS_SUBDIR)
    p = os.path.join(config.run_dir, SETTING_PLOTS_SUBDIR)

    print("\nCreating directories:\n")
    print("\t" + config.run_dir)
    print("\t" + d)
    print("\t" + p)

    os.makedirs(config.run_dir, exist_ok=False)
    os.makedirs(d, exist_ok=False)
    os.makedirs(p, exist_ok=False)



