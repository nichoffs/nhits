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

class MultiInterpolater(nn.Module):
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
        # reshape x to (batch_size, 1, samples, feature_dim)
        x = x.view(x.shape[0], 1, -1, x.shape[-1])
        # interpolate, maintaining the feature dimension
        x = F.interpolate(x, (self.forecast_size, x.shape[-1]), mode=self.interpolation_mode)
        # reshape back to (batch_size, samples, feature_dim)
        x = x.view(x.shape[0], -1, x.shape[-1])
        return x

    def forward(self, knots):
        forecast = self._interpolate(knots)
        return forecast
