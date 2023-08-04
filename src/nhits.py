import torch
import torch.nn as nn
from nhits_block import NHiTS_Block
from torch.utils.data import DataLoader
from dataset.dataset import UnivariateTSDataset
from interpolater import Interpolater
import argparse
import yaml

device = "mps"

with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--lookback_horizon", type=int, default=config.get("lookback_horizon")
)
parser.add_argument("--forecast_size", type=int, default=config.get("forecast_size"))
parser.add_argument("--batch_size", type=int, default=config.get("batch_size"))
parser.add_argument(
    "--num_nhits_blocks", type=int, default=config.get("num_nhits_blocks")
)
parser.add_argument("--input_size", type=int, default=config.get("input_size"))
parser.add_argument("--mlp_layer_num", type=int, default=config.get("mlp_layer_num"))
parser.add_argument("--hidden_size", type=int, default=config.get("hidden_size"))
parser.add_argument(
    "--pooling_kernel_size",
    nargs="+",
    type=int,
    default=config.get("pooling_kernel_size"),
)
parser.add_argument(
    "--downsampling_ratios",
    nargs="+",
    type=int,
    default=config.get("downsampling_ratios"),
)
parser.add_argument("--dropout_prob", type=float, default=config.get("dropout_prob"))
parser.add_argument("--num_samples", type=int, default=config.get("num_samples"))

# Parse arguments
args = parser.parse_args()

bc_coefs_size = [
    args.lookback_horizon // downsampling_ratio
    for downsampling_ratio in args.downsampling_ratios
]
fc_coefs_size = [
    args.forecast_size // downsampling_ratio
    for downsampling_ratio in args.downsampling_ratios
]


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
        num_nhits_blocks=3,
        mlp_layer_num=2,
        hidden_size=512,
        pooling_kernel_size=[2, 2, 2],
        forecast_size=24,
        pooling_mode="max",
        dropout_prob=0.0,
        num_samples=1000,
        downsampling_ratios=[24, 12, 1],
        batch_size=128,
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
        self.bc_predictors = Interpolater("linear", lookback_horizon)
        self.fc_predictors = Interpolater("linear", forecast_size)

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
