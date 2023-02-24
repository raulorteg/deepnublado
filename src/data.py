"""
This file contains the main data class(es)
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class DeepNubladoData(Dataset):

    def __init__(self, parameters, run_outcomes,
                 emission_lines, continuum_data=None,
                 split='train',
                 split_fraction=(0.8, 0.1, 0.1)):

        if sum(split_fraction) != 1:
            raise Exception(f"Fractions of train | val | test should add up to 1.0")

        train_fraction, val_fraction, test_fraction = split_fraction

        n_samples = parameters.shape[0]

        if split == 'train':
            begin = 0
            last = int(train_fraction * n_samples)

        if split == 'val':
            begin = int(train_fraction * n_samples)
            last = int((train_fraction + val_fraction) * n_samples)

        if split == 'test':
            begin = int((train_fraction + val_fraction) * n_samples)
            last = -1

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