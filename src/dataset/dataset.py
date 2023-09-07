from torch.utils.data import Dataset
import torch

class NHiTS_Dataset(Dataset):
    def __init__(self, series, backcast_size, forecast_size):
        self.series = series
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def __len__(self):
        return len(self.series) - self.backcast_size - self.forecast_size + 1

    def __getitem__(self, idx):
        input_data = self.series[idx: idx + self.backcast_size]
        target_data = self.series[idx + self.backcast_size: idx + self.backcast_size + self.forecast_size]
       
        # Convert to PyTorch tensors
        input_data = torch.Tensor(input_data.values)
        target_data = torch.Tensor(target_data.values)

        return input_data, target_data
