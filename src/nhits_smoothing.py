import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter


class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, x):
        return self.layer(x)
    
class Basis(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        pass

class InterpolationBasis(Basis):
    def __init__(self, backcast_size, forecast_size):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, bcc, fcc):
        backcast = F.interpolate(bcc.unsqueeze(1), self.backcast_size).squeeze()
        forecast = F.interpolate(fcc.unsqueeze(1), self.forecast_size).squeeze()
        return backcast, forecast



class NHiTS_Block_Redesign(nn.Module):
    def __init__(
        self,
        pooling_size,
        pooling_mode,
        mlp_hidden_size,
        mlp_num_layers,
        bcc_size,
        fcc_size,
        backcast_size,
    ):
        super().__init__()
        assert pooling_mode in ["max", "avg"], "pooling_mode must be max or avg"

        self.bcc_size = bcc_size
        self.fcc_size = fcc_size

        self.pooling = (
            nn.MaxPool1d(kernel_size=pooling_size, stride=pooling_size)
            if pooling_mode == "max"
            else nn.AvgPool1d(kernel_size=pooling_size, stride=pooling_size)
        )
        self.mlps = nn.Sequential(
            nn.Linear(backcast_size // pooling_size, mlp_hidden_size),
            *(MLP(mlp_hidden_size) for _ in range(mlp_num_layers)),
        )

        self.bcc_linear = nn.Linear(mlp_hidden_size, bcc_size)
        self.fcc_linear = nn.Linear(mlp_hidden_size, fcc_size)

    def forward(self, x):
        x = self.pooling(x)
        x = self.mlps(x)

        bcc = self.bcc_linear(x)
        fcc = self.fcc_linear(x)

        return bcc, fcc


class NHiTS_Stack_Redesign(nn.Module):
    def __init__(
        self,
        num_blocks,
        pooling_size,
        pooling_mode,
        mlp_hidden_size,
        mlp_num_layers,
        bcc_size,
        fcc_size,
        backcast_size,
        forecast_size,
    ):
        super().__init__()
        
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        
        self.basis = InterpolationBasis(backcast_size=backcast_size,forecast_size=forecast_size)

        self.blocks = nn.ModuleList(
            [
                NHiTS_Block_Redesign(
                    pooling_size=pooling_size,
                    pooling_mode=pooling_mode,
                    mlp_hidden_size=mlp_hidden_size,
                    mlp_num_layers=mlp_num_layers,
                    bcc_size=bcc_size,
                    fcc_size=fcc_size,
                    backcast_size=backcast_size,
                )
                for block in range(num_blocks)
            ]
        )

    def forward(self, x):
        res_stream = x
        outputs = []
        for block in self.blocks:
            bcc, fcc = block(res_stream)
            
            # backcast,forecast = self.basis(bcc.unsqueeze(1), fcc.unsqueeze())
            # print(f'{bcc.shape=}, {fcc.shape=}')
            
            backcast = F.interpolate(bcc.unsqueeze(1), self.backcast_size).squeeze()
            forecast = F.interpolate(fcc.unsqueeze(1), self.forecast_size).squeeze()
            # print(f'{backcast.shape=}, {forecast.shape=}')
            outputs.append(forecast)
            res_stream = res_stream - backcast

        return torch.sum(torch.stack(outputs), dim=0)


class NHiTS_Redesign(nn.Module):
    def __init__(
        self,
        stacks_num_layers,
        backcast_size,
        forecast_size,
        num_blocks,
        pooling_size,
        pooling_mode,
        mlp_hidden_size,
        mlp_num_layers,
        bcc_size,
        fcc_size
    ):
        super().__init__()
        
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size

        self.stacks = nn.ModuleList(
            [
                NHiTS_Stack_Redesign(
                    backcast_size=backcast_size,
                    forecast_size=forecast_size,
                    num_blocks=num_blocks,
                    pooling_size=pooling_size[i],
                    pooling_mode=pooling_mode,
                    mlp_hidden_size=mlp_hidden_size,
                    mlp_num_layers=mlp_num_layers,
                    bcc_size=bcc_size[i],
                    fcc_size=fcc_size[i],
                )
                for i in range(stacks_num_layers)
            ]
        )

    def forward(self, x):
        res_stream = x
        outputs = []
        for stack in self.stacks:
            outputs.append(stack(res_stream))
        return torch.sum(torch.stack(outputs), dim=0)

    def fit(self, dataloader, loss_func, lr, optim, num_epochs, logging=False):
        optimizer = optim(self.parameters(), lr=lr)

        for epoch in range(num_epochs):
            for batch_idx, (data, targets) in enumerate(dataloader):
                # Forward pass
                outputs = self.forward(data)
                # Compute loss
                loss = loss_func(outputs, targets)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                writer.add_scalar("Loss/train", loss, epoch)
                optimizer.step()

                if (batch_idx + 1) % 100 == 0:  # Print every 100 batches
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item()}"
                    )

    def predict(self, data_loader):
        self.eval()
        predictions = []
        with torch.no_grad():
            for data in data_loader:
                print(type(data))
                output = self.forward(data)
                predictions.append(output)
        return torch.stack(predictions)
    
    def predict_single(self, sample):
        assert sample.shape[-1]==self.backcast_size and len(sample.shape==2), 'Must have rank 2 and final shape dim be len backcast_size-> (1, backcast_size)'
        self.eval()
        predictions = []
        with torch.no_grad():
            output = self.forward(sample)
            predictions.append(output)
        return torch.stack(predictions)
    
    def visualize_predictions(self, sample):
        
        
        plt.figure(figsize=(10, 5))
        plt.plot(actual, label="Actual", color="blue")
        plt.plot(predictions, label="Predictions", color="red", linestyle="--")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.title("Predictions vs Actual")
        plt.show()

if __name__ == "__main__":
    writer = SummaryWriter()

    model = NHiTS_Redesign(3,500,100,3,[32,8,1], [])