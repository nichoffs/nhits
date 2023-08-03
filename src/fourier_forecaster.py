import torch
import torch.nn as nn
from base_forecaster import Forecaster

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
        
        # Dimensions of x should be (B, num_samples)
        # Dimensions of amps, freqs, phases should be (B, num_freqs, 1) or (1, num_freqs, 1)

        # Generate a tensor of sine waves based on the input frequencies and phases
        sine_waves = torch.sin(freqs.unsqueeze(2) * t + phases.unsqueeze(2))

        # Multiply the amplitudes with the sine waves
        out = torch.sum(amps.unsqueeze(2) * sine_waves, dim=1)
        return out
