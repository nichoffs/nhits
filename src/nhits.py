import argparse

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from dataset.dataset import UnivariateTSDataset
from interpolater import Interpolater
from nhits_block import NHiTS_Block


class NHiTS(nn.Module):
    """
    This script provides an implementation of the NHiTS) architecture for time-series prediction.

    All values default to values in config.yaml.

    args:
        lookback_horizon : Number of past observations to consider
        forecast_size : Number of future observations to forecast
        batch_size : Size of each batch
        num_nhits_blocks : Number of NHiTS blocks in the architecture
        mlp_layer_num : Number of layers in the MLP
        hidden_size : Size of hidden layers in the model
        pooling_kernel_size : Size of the pooling kernel
        downsampling_ratios : Ratios for downsampling
        dropout_prob : Dropout probability for the dropout layer
        num_samples : Number of samples to consider
    """

    def __init__(
        self,
        lookback_horizon=120,
        forecast_size=24,
        batch_size=128,
        num_nhits_blocks=3,
        mlp_layer_num=2,
        hidden_size=512,
        pooling_kernel_size=[2, 2, 2],
        downsampling_ratios=[4, 2, 1],
        dropout_prob=0.0,
        pooling_mode="max",
    ):
        super().__init__()

        bc_coefs_size = [
            lookback_horizon // downsampling_ratio
            for downsampling_ratio in downsampling_ratios
        ]
        fc_coefs_size = [
            forecast_size // downsampling_ratio
            for downsampling_ratio in downsampling_ratios
        ]

        self.lookback_horizon = lookback_horizon
        self.blocks = nn.ModuleList(
            [
                NHiTS_Block(
                    lookback_horizon,
                    mlp_layer_num,
                    hidden_size,
                    bc_coefs_size[i],
                    fc_coefs_size[i],
                    pooling_kernel_size[i],
                    forecast_size,
                    forecaster_type="linear",
                    pooling_mode="max",
                    dropout_prob=0.0,
                )
                for i in range(num_nhits_blocks)
            ]
        )
        self.bc_predictors = Interpolater(lookback_horizon, 'linear')
        self.fc_predictors = Interpolater(forecast_size, 'linear')

    def forward(self, x):
        res_stream = x
        forecasts = []

        for block in self.blocks:
            bc_coefs, fc_coefs = block(res_stream)
            bc = self.bc_predictors(bc_coefs)
            fc = self.fc_predictors(fc_coefs)
            forecasts.append(fc)
            res_stream = res_stream - bc
        return torch.sum(torch.stack(forecasts), dim=0)
