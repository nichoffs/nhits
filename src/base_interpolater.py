import torch
import torch.nn as nn
import torch.nn.functional as F

class Interpolater(nn.Module):

    def __init__(self, interpolation_mode, forecast_size):
        super().__init__()
        self.interpolation_mode = interpolation_mode
        self.forecast_size=forecast_size
    
    def _interpolate(self, x):
        #knots.shape = (B, num_knots)--->(B, num_knots,1)
        x=x.unsqueeze(1)
        return F.interpolate(x, self.forecast_size, mode='nearest').squeeze()
    
    def forward(self, knots):
        forecast = self._interpolate(knots)
        return forecast
