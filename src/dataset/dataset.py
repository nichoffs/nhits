import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

class UnivariateTSDataset(torch.utils.data.Dataset):
    def __init__(self, path, col, random_seed=42, lookback_horizon=120, forecast_horizon=24):
        self.random_seed = random_seed
        self.lookback_horizon = lookback_horizon
        self.forecast_horizon = forecast_horizon
        
        # Load and preprocess data
        self.data = pd.read_csv(path)
        self.data = self.data[col]
        self.data = self.data.values.astype(np.float32)  # Convert to numpy array and ensure dtype is float32

        # Make sure the random seed is set for reproducibility
        np.random.seed(self.random_seed)
        
    def __len__(self):
        # Subtract sample_size because for each point, we need to return the next sample_size points as well
        return len(self.data) - (self.lookback_horizon + self.forecast_horizon)

    def __getitem__(self, index):
        # Get item at index and the next sample_size items
        item = self.data[index : index + self.lookback_horizon]
        target = self.data[index + self.lookback_horizon : index + self.lookback_horizon + self.forecast_horizon]
        
        
        # Convert item to a PyTorch tensor and return
        return torch.from_numpy(item), torch.from_numpy(target)
