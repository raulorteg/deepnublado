"""
This file contains the model classes used for our experiments
"""

from torch import nn
from settings import DEEPNUBLADO_INPUTS, DEEPNUBLADO_REGRESSOR_OUTPUTS


class MLP1(nn.Module):
    """
    A simple first model that maps input parameters to emission lines.

    Use of dropout and batch norm is toggled via config object.
    """

    def __init__(self, conf):
        super(MLP1, self).__init__()

        def block(features_in: int,
                  features_out: int,
                  batch_norm: bool = conf.batch_norm,
                  dropout: bool = conf.dropout):

            layers = [nn.Linear(features_in, features_out)]

            if batch_norm:
                layers.append(nn.BatchNorm1d(features_out))

            if dropout:
                layers.append(nn.Dropout(conf.dropout_value))

            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers

        self.model = nn.Sequential(
            *block(len(DEEPNUBLADO_INPUTS), 64, batch_norm=False, dropout=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, len(DEEPNUBLADO_REGRESSOR_OUTPUTS))
        )

    def forward(self, parameters):

        return self.model(parameters)
