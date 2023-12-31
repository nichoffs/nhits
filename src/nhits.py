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
from dataset.dataset import NHiTS_Dataset
from torch.utils.data import DataLoader
import pandas as pd
import io
import numpy as np
from PIL import Image
from torchvision import transforms
class ForecastLogger:
    def __init__(self, writer):
        self.writer = writer
        self.global_step = 0

    def log_forecasts(self, full_backcast, knots, image_title='Plot'):
        plt.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
                
        image = Image.open(buf)
        image_tensor = transforms.ToTensor()(image)

        self.writer.add_image(image_title, image_tensor, self.global_step, dataformats='CHW')

        # Close the buffer and plot
        buf.close()
        plt.close()


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
        backcast_size
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
        forecast_logger
    ):
        super().__init__()
        
        self.forecast_logger = forecast_logger
        
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        
        self.basis = InterpolationBasis(backcast_size=backcast_size,forecast_size=forecast_size)

        self.blocks = nn.ModuleList(
            [
                NHiTS_Block_Redesign(
                    pooling_size=pooling_size[i],
                    pooling_mode=pooling_mode,
                    mlp_hidden_size=mlp_hidden_size,
                    mlp_num_layers=mlp_num_layers,
                    bcc_size=bcc_size,
                    fcc_size=fcc_size,
                    backcast_size=backcast_size
                    
                )
                for i in range(num_blocks)
            ]
        )

    def forward(self, x):
        res_stream = x
        outputs = []
        for block in self.blocks:
            bcc, fcc = block(res_stream)
            
            
            backcast = F.interpolate(bcc.unsqueeze(1), self.backcast_size).squeeze()
            self.forecast_logger.log_forecasts()
            forecast = F.interpolate(fcc.unsqueeze(1), self.forecast_size).squeeze()
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
        fcc_size,
        writer,
        logging=True
    ):
        super().__init__()
        self.logging = logging
        if logging:
            self.writer = SummaryWriter()
            self.forecast_logger = ForecastLogger(writer)
        
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        # TODO: FIT LOGGING INITIALIZATION
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
                    forecast_logger=self.forecast_logger
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

    def fit(self, dataloader, loss_func, lr, optim, num_epochs):

        
        optimizer = optim(self.parameters(), lr=lr)

        counter=0
        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(dataloader):
                # Forward pass
                output = self.forward(data)
                # Compute loss
                loss = loss_func(output, target)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                
                optimizer.step()
                counter += 1

                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item()}"
                )
                if self.writer and self.forecast_logger:
                        self.writer.add_scalar("Loss/train", loss, counter)
                        self.forecast_logger.log_forecasts(output, target, f'Forecast_{counter}')

    def predict(self, data_loader):
        self.eval()
        predictions = []
        with torch.no_grad():
            for data in data_loader:
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

    model = NHiTS_Redesign(3,480,96,3,[[128,64,32],[32,16,8],[4,2,1]], 'max', 128, 3, [16, 64, 128], [8,32,64], writer=writer, True)
    cols = pd.read_csv('./data/ETTm2.csv').columns 
    for col in cols:
            
        df = pd.read_csv('./data/ETTm2.csv')['MULL']
        
        dataset = NHiTS_Dataset(df, 480, 96)
        
        dl = DataLoader(dataset, 128, shuffle=True)  
        
        model.fit(dl, nn.MSELoss(), .001, torch.optim.AdamW, 10, True)