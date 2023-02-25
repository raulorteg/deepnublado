"""
Main run script for our DeepNublado experiments
"""
import argparse
import os
import torch

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

from settings import SETTING_NUM_EPOCHS, \
    SETTING_BATCH_SIZE, \
    SETTING_LEARNING_RATE, \
    SETTING_P_DROPOUT

# -----------------------------------------------------------------
#  CUDA available?
# -----------------------------------------------------------------
if torch.cuda.is_available():
    cuda = True
    device = torch.device("cuda")
    FloatTensor = torch.cuda.FloatTensor
else:
    cuda = False
    device = torch.device("cpu")
    FloatTensor = torch.FloatTensor


def main(config):
    """
    Main driver function for running our experiments

    :param config: argparse object
    :return: nothing
    """
    pass


# -----------------------------------------------------------------
#  The following is executed when the script is run
# -----------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="DeepNublado - Cloudy with deep neural networks"
    )

    # arguments for data handling
    parser.add_argument(
        "--csv_dir",
        type=str,
        help="Path to cvs directory",
        required=True
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="output",
        help="Path to output directory, used for all plots and data products, default: ./output/"
    )

    # network optimisation
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=SETTING_NUM_EPOCHS,
        help="number of training epochs"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=SETTING_BATCH_SIZE,
        help=f"size of the batches, default={SETTING_BATCH_SIZE}"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=SETTING_LEARNING_RATE,
        help=f"adam: learning rate, default={SETTING_LEARNING_RATE} "
    )

    parser.add_argument(
        "--dropout_value",
        type=float,
        default=SETTING_P_DROPOUT,
        help=f"dropout probability, default={SETTING_P_DROPOUT} "
    )

    # BN on / off
    parser.add_argument(
        "--batch_norm",
        dest="batch_norm",
        action="store_true",
        help="use batch normalisation in network"
    )
    parser.add_argument(
        "--no-batch_norm",
        dest="batch_norm",
        action="store_false",
        help="do not use batch normalisation in network (default)"
    )
    parser.set_defaults(batch_norm=False)

    # dropout on / off
    parser.add_argument(
        "--dropout",
        dest="dropout",
        action="store_true",
        help="use dropout regularisation in network"
    )
    parser.add_argument(
        "--no-dropout",
        dest="dropout",
        action="store_false",
        help="do not use dropout regularisation in network (default)"
    )
    parser.set_defaults(dropout=False)

    # run analysis / plotting  routines after training?
    parser.add_argument(
        "--analysis",
        dest="analysis",
        action="store_true",
        help="automatically generate some plots (default)"
    )
    parser.add_argument(
        "--no-analysis",
        dest="analysis",
        action="store_false",
        help="do not run analysis"
    )
    parser.set_defaults(analysis=True)

    my_config = parser.parse_args()

    my_config.out_dir = os.path.abspath(my_config.out_dir)
    my_config.csv_dir = os.path.abspath(my_config.csv_dir)

    main(my_config)

