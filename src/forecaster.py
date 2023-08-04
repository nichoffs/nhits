import torch
import torch.nn as nn


class Forecaster(nn.Module):
    """Forecaster base class

    Args:
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, coefs):
        pass

class FourierForecaster(Forecaster):
    """Creates forecasts for num_freqs sine waves based on amps,freqs,phases and sums

    Args:
        Forecaster (_type_): _description_
    """

    def __init__(self, num_freqs, time_s):
        super().__init__()
        self.num_freqs = num_freqs
        self.time_s = time_s

    def forward(self, lookback_horizon, amps, freqs, phases):
        t = torch.linspace(0, self.time_s, lookback_horizon)
        sine_waves = torch.sin(freqs.unsqueeze(2) * t + phases.unsqueeze(2))
        out = torch.sum(amps.unsqueeze(2) * sine_waves, dim=1)
        return out
