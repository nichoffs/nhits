import argparse

import torch
import torch.nn as nn
import typer
import yaml

from nhits import NHiTS

device = torch.device("mps")

with open("./config.yaml", "r") as f:
    config = yaml.safe_load(f)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--lookback_horizon", type=int, default=config.get("lookback_horizon")
)
parser.add_argument("--forecast_size", type=int, default=config.get("forecast_size"))
parser.add_argument("--batch_size", type=int, default=config.get("batch_size"))
parser.add_argument(
    "--num_nhits_blocks", type=int, default=config.get("num_nhits_blocks")
)
parser.add_argument("--mlp_layer_num", type=int, default=config.get("mlp_layer_num"))
parser.add_argument("--hidden_size", type=int, default=config.get("hidden_size"))
parser.add_argument(
    "--pooling_kernel_size",
    nargs="+",
    type=int,
    default=config.get("pooling_kernel_size"),
)
parser.add_argument(
    "--downsampling_ratios",
    nargs="+",
    type=int,
    default=config.get("downsampling_ratios"),
)
parser.add_argument("--dropout_prob", type=float, default=config.get("dropout_prob"))

# Parse arguments
args = parser.parse_args()

bc_coefs_size = [
    args.lookback_horizon // downsampling_ratio
    for downsampling_ratio in args.downsampling_ratios
]
fc_coefs_size = [
    args.forecast_size // downsampling_ratio
    for downsampling_ratio in args.downsampling_ratios
]

# Import needed libraries
from torch.utils.data import DataLoader

from dataset.dataset import UnivariateTSDataset

# Initialize your dataset
dataset = UnivariateTSDataset(
    path='./data/ETTm2.csv', 
    col='MULL', # replace 'desired_column' with the column name you want
    lookback_horizon=args.lookback_horizon, 
    forecast_horizon=args.forecast_size
)

# Initialize a DataLoader with your dataset
batch_size = args.batch_size
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Now you can use the dataloader in your training loop



def train_model(lookback_horizon, forecast_size, batch_size, num_nhits_blocks, mlp_layer_num, hidden_size, pooling_kernel_size, downsampling_ratios, dropout_prob):
    model = NHiTS(
        lookback_horizon=lookback_horizon,
        forecast_size=forecast_size,
        num_nhits_blocks=num_nhits_blocks,
        mlp_layer_num=mlp_layer_num,
        hidden_size=hidden_size,
        pooling_kernel_size=pooling_kernel_size,
        downsampling_ratios=downsampling_ratios,
        dropout_prob=dropout_prob
    ).to(device)
    
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(config['epochs']):
        for batch in dataloader: # assuming dataloader is previously defined
            # Zero the gradients
            optimizer.zero_grad()

            # Prepare the inputs and targets
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

             # Forward pass
            outputs = model(inputs)

             # Compute loss
            loss = criterion(outputs, targets)

             # Backward pass
            loss.backward()

             # Update weights
            optimizer.step()
        print(loss)
if __name__ == "__main__":
    print(vars(args))
    train_model(**vars(args))