"""
This file contains the main data class(es)
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from settings import SETTING_TRAIN_FRACTION, SETTING_VAL_FRACTION, SETTING_TEST_FRACTION


class DeepNubladoData(Dataset):

    def __init__(self, parameters, run_outcomes,
                 emission_lines, continuum_data=None,
                 split: str = 'train',
                 split_fraction: tuple = (SETTING_TRAIN_FRACTION,
                                          SETTING_VAL_FRACTION,
                                          SETTING_TEST_FRACTION)
                 ):

        if sum(split_fraction) != 1:
            raise Exception(f"DeepNubladoData init: Fractions of train | val | test should add up to 1.0")

        train_fraction, val_fraction, test_fraction = split_fraction

        n_samples = parameters.shape[0]

        if split == 'train':
            begin = 0
            last = int(train_fraction * n_samples)

        elif split == 'val':
            begin = int(train_fraction * n_samples)
            last = int((train_fraction + val_fraction) * n_samples)

        elif split == 'test':
            begin = int((train_fraction + val_fraction) * n_samples)
            last = -1
        else:
            raise Exception(f"DeepNubladoData init:split should be either train, val, or test")

        self.parameters = torch.from_numpy(parameters[begin:last]).type(torch.FloatTensor)
        self.run_outcomes = torch.from_numpy(run_outcomes[begin:last]).type(torch.FloatTensor)
        self.emission_lines = torch.from_numpy(emission_lines[begin:last]).type(torch.FloatTensor)

        if continuum_data:
            self.continuum_data = torch.from_numpy(continuum_data[begin:last]).type(torch.FloatTensor)

    def __len__(self):
        return self.parameters.shape[0]

    # def __getitem__(self, index):
    #
    #     out = []
    #     out.append(self.parameters[index])
    #     out.append(self.run_outcomes[index])
    #
    #     for i in range(self.emission_lines.shape[1]):
    #         out.append(self.emission_lines[index, i])
    #
    #     if self.continuum_data:
    #         for i in range(self.continuum_data.shape[1]):
    #             out.append(self.continuum_data[index, i])
    #
    #     return tuple(out)