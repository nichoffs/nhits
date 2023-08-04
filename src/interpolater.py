import torch
import torch.nn as nn
import torch.nn.functional as F


class Interpolater(nn.Module):
    """Interpolater - base class for interpolation/upsampling

    Args:
        interpolation_mode (str): interpolation mode for F.interpolate
        forecast_size (int): number of samples to interpolate into
    """

    def __init__(self, forecast_size=120, interpolation_mode="linear"):
        super().__init__()
        self.interpolation_mode = interpolation_mode
        self.forecast_size = forecast_size

    def _interpolate(self, x):
        x = x.unsqueeze(1)
        return F.interpolate(x, self.forecast_size, mode=self.interpolation_mode).squeeze()

    def forward(self, knots):
        forecast = self._interpolate(knots)
        return forecast
