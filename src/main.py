"""
Main run script for our DeepNublado experiments
"""
import argparse
import os
import copy
import logging
import torch
import numpy as np


from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

from models import MLP1
from setup import setup_main
from data import DeepNubladoData
from utils import \
    utils_rescale_inputs, \
    utils_de_transform_line_data, \
    utils_save_model, \
    utils_save_loss, \
    utils_save_config_to_file, \
    utils_save_config_to_log, \
    utils_save_test_data

from settings import \
    SETTING_MAIN_OUTPUT_DIR, \
    SETTING_NUM_EPOCHS, \
    SETTING_BATCH_SIZE, \
    SETTING_LEARNING_RATE, \
    SETTING_P_DROPOUT, \
    DEEPNUBLADO_REGRESSOR_OUTPUTS, \
    SETTING_DATA_PRODUCTS_SUBDIR, \
    SETTING_TEST_FREQ, \
    SETTING_NORMALISE_INPUTS, \
    SETTING_TRANSFORM_LINE_DATA

from analysis import analysis_main
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


# -----------------------------------------------------------------
#  loss function(s)
# -----------------------------------------------------------------
def loss_function(gen_x, real_x, config):
    """
    Computes the loss function(s).

    Note: This is its own function in case we want to add complexity here later

    :param gen_x: inferred data
    :param real_x: simulated data (ground truth)
    :param config: user config (in case it's needed later)
    :return:
    """

    loss = F.mse_loss(input=gen_x,
                      target=real_x.view(-1, len(DEEPNUBLADO_REGRESSOR_OUTPUTS)),
                      reduction='mean')

    return loss


# -----------------------------------------------------------------
#  Training
# -----------------------------------------------------------------
def train_model(model, optimizer, train_loader, config):
    """
    This function trains the network for one epoch.

    :param model: current model state
    :param optimizer: optimizer object to perform the back-propagation
    :param train_loader: data loader containing training data
    :param config: config object with user supplied parameters
    :return: Averaged training loss. No need to return the model as the optimizer modifies it inplace.
    """

    if cuda:
        model.cuda()

    model.train()
    train_loss = 0

    for batch_idx, (inputs, emission_lines) in enumerate(train_loader):

        # configure data
        real_lines = Variable(emission_lines.type(FloatTensor))
        real_inputs = Variable(inputs.type(FloatTensor))

        # zero the gradients on each iteration
        optimizer.zero_grad()

        # generate a batch of lines
        gen_lines = model(real_inputs)

        # estimate loss
        loss = loss_function(gen_lines, real_lines, config)

        train_loss += loss.item()    # average loss per batch

        # back propagation
        loss.backward()
        optimizer.step()

    average_loss = train_loss / len(train_loader)   # divide by number of batches (!= batch size)

    return average_loss  # float


# -----------------------------------------------------------------
#   evaluate model with test or validation set
# -----------------------------------------------------------------
def evaluate_model(current_epoch: int, data_loader, model, path, config,
                   save_results=False, best_model=False):
    """
    function runs the given dataset through the model, returns mse_loss,
    and (optionally) saves the results as well as ground truth to file.

    Args:
        current_epoch: current epoch
        data_loader: data loader used for the inference, most likely the test set
        path: path to output directory
        model: current model state
        config: config object with user supplied parameters
        save_results: flag to save generated profiles locally (default: False)
        best_model: flag for testing on best model
    """

    if save_results:
        print(F"\033[94m\033[1mTesting the network now at epoch {current_epoch} \033[0m")

    if cuda:
        model.cuda()

    if save_results:
        lines_gen_all = torch.tensor([], device=device)
        lines_true_all = torch.tensor([], device=device)
        inputs_true_all = torch.tensor([], device=device)
        # getting ground truth data here, so we don't have to worry about
        # randomisation of the samples.

    model.eval()

    loss = 0.0

    with torch.no_grad():
        for i, (inputs, emission_lines) in enumerate(data_loader):

            # configure input
            lines_true = Variable(emission_lines.type(FloatTensor))
            inputs = Variable(inputs.type(FloatTensor))

            # inference
            lines_gen = model(inputs)

            loss += loss_function(lines_true, lines_gen, config)

            if save_results:
                # collate data from different batches
                lines_gen_all = torch.cat(tensors=(lines_gen_all, lines_gen), dim=0)
                lines_true_all = torch.cat(tensors=(lines_true_all, lines_true), dim=0)
                inputs_true_all = torch.cat(tensors=(inputs_true_all, inputs), dim=0)

    # mean of computed losses
    loss = loss / len(data_loader)

    if save_results:
        # move data to CPU, re-scale inputs, and write everything to file
        lines_gen_all = lines_gen_all.cpu().numpy()
        lines_true_all = lines_true_all.cpu().numpy()
        inputs_true_all = inputs_true_all.cpu().numpy()

        if SETTING_NORMALISE_INPUTS:
            inputs_true_all = utils_rescale_inputs(parameters=inputs_true_all)

        if SETTING_TRANSFORM_LINE_DATA:
            lines_gen_all = utils_de_transform_line_data(emission_lines=lines_gen_all)
            lines_true_all = utils_de_transform_line_data(emission_lines=lines_true_all)

        if best_model:
            prefix = 'best'
        else:
            prefix = 'test'

        utils_save_test_data(
            parameters=inputs_true_all,
            lines_true=lines_true_all,
            lines_gen=lines_gen_all,
            path=path,
            epoch=current_epoch,
            prefix=prefix
            )


    return loss.item()


# -----------------------------------------------------------------
#  Main driver function
# -----------------------------------------------------------------
def main(config):
    """
    Main driver function for running our experiments

    :param config: argparse object
    :return: nothing
    """

    train_loader, val_loader, test_loader = setup_main(config)

    # select model and prep optimizer
    model = MLP1(conf=config)

    if cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.lr,
                                 betas=(0.9, 0.999)
                                 )

    # variables for bookkeeping
    train_loss_array = val_loss_array = np.empty(0)
    best_model = copy.deepcopy(model)
    best_loss = np.inf
    best_epoch = 1
    n_epoch_without_improvement = 0
    stopped_early = False
    epochs_trained = -1
    data_products_path = os.path.join(config.run_dir, SETTING_DATA_PRODUCTS_SUBDIR)

    # main training loop
    print("\033[96m\033[1m\nTraining starts now\033[0m")
    for epoch in range(1, config.n_epochs + 1):

        train_loss = train_model(model, optimizer, train_loader, config)

        train_loss_array = np.append(train_loss_array, train_loss)

        val_loss = evaluate_model(current_epoch=epoch,
                                  data_loader=val_loader,
                                  model=model,
                                  path=data_products_path,
                                  config=config,
                                  save_results=False,
                                  best_model=False
                                  )

        val_loss_array = np.append(val_loss_array, val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            n_epoch_without_improvement = 0
        else:
            n_epoch_without_improvement += 1

        print(f"[Epoch {epoch:4}/{config.n_epochs:4}] "
              f"  [Training loss: {train_loss:8.4e}] "
              f"  [Validation loss: {val_loss:8.4e}]"
              f"  [Best epoch: {best_epoch:4}]"
              )

        # check for testing criterion
        if epoch % SETTING_TEST_FREQ == 0 or epoch == config.n_epochs:

            _ = evaluate_model(current_epoch=epoch,
                               data_loader=test_loader,
                               model=model,
                               path=data_products_path,
                               config=config,
                               save_results=True,
                               best_model=False
                               )

    print("\033[96m\033[1m\nTraining complete\033[0m\n")
    logging.info(f"Training complete. Best epoch is N={best_epoch}")

    # evaluate the best model
    _ = evaluate_model(current_epoch=best_epoch,
                       data_loader=test_loader,
                       model=best_model,
                       path=data_products_path,
                       config=config,
                       save_results=True,
                       best_model=True
                       )

    # Save the best model and loss functions
    utils_save_model(state=best_model.state_dict(),
                     path=data_products_path,
                     n_epoch=best_epoch,
                     best_model=True)

    utils_save_loss(loss_array=train_loss_array,
                    path=data_products_path,
                    n_epoch=config.n_epochs,
                    prefix='train'
                    )
    utils_save_loss(loss_array=val_loss_array,
                    path=data_products_path,
                    n_epoch=config.n_epochs,
                    prefix='val'
                    )

    # Save some results to config object for later use
    config.best_epoch = best_epoch
    config.best_val = best_loss
    config.cuda_used = cuda

    utils_save_config_to_file(config)
    utils_save_config_to_log(config)

    if config.analysis:
        print("\n\033[96m\033[1m\nRunning analysis\033[0m\n")
        analysis_main(config)

    # TODO: add early stopping without crashing
    # TODO: add model selection once we have more than one
    # TODO: add continuum data


# -----------------------------------------------------------------
#  The following is executed when the script is run
# -----------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="DeepNublado - Cloudy with deep neural networks"
    )

    # arguments for data handling
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to (CSV) data directory",
        required=True
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=SETTING_MAIN_OUTPUT_DIR,
        help=f"Path to output directory, used for all plots and data products, default: {SETTING_MAIN_OUTPUT_DIR}"
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
    my_config.data_dir = os.path.abspath(my_config.data_dir)

    main(my_config)
