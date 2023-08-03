import torch.nn as nn
from fourier_forecaster import FourierForecaster
from base_interpolater import Interpolater


class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, x):
        return self.layer(x)


class NHiTS_Block(nn.Module):
    """NHiTS Block

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        lookback_horizon,
        mlp_layer_num,
        hidden_size,
        bc_coefs_size,
        fc_coefs_size,
        pooling_kernel_size,
        forecast_size,
        forecaster_type="linear",
        pooling_mode="max",
        dropout_prob=0.0,
    ):
        super().__init__()

        assert forecaster_type in [
            "fourier",
            "linear",
        ], "forecaster must be one of 'fourier', 'linear'"
        assert pooling_mode in ["max", "avg"], "pooling_mode must be max or avg"

        pooling = nn.MaxPool1d if pooling_mode == "max" else nn.AvgPool1d
        self.pooling_kernel = pooling(
            kernel_size=pooling_kernel_size, stride=pooling_kernel_size
        )

        self.mlps = nn.ModuleList([MLP(hidden_size) for _ in range(mlp_layer_num)])

        self.encode_layer = nn.Linear(
            lookback_horizon // pooling_kernel_size, hidden_size, bias=False
        )

        self.backcast_fc = nn.Linear(hidden_size, bc_coefs_size)
        self.forecast_fc = nn.Linear(hidden_size, fc_coefs_size)

    def fourier_coefs(self, coefs):
        return coefs.view(-1, 10, 3).chunk(3, dim=2)

    def forward(self, x):
        x = self.pooling_kernel(x)
        x = self.encode_layer(x)
        for mlp in self.mlps:
            x = mlp(x)
        backast_coefs = self.backcast_fc(x)
        forecast_coefs = self.forecast_fc(x)

        return backast_coefs, forecast_coefs
