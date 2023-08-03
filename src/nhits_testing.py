from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from nhits import NHiTS
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def create_sequences(input_data, tw, fh):
    sequences = []
    L = len(input_data)
    for i in range(L - tw - fh + 1):
        train_seq = input_data[i : i + tw]
        train_label = input_data[i + tw : i + tw + fh]
        sequences.append((train_seq, train_label))
    return sequences


def prepare_data(df, column, lookback_horizon, forecast_horizon, batch_size):
    # Extract the time series
    ts = df[column].values.reshape(-1, 1)
    print(df.columns)

    # Normalize the data
    scaler = StandardScaler()
    ts = scaler.fit_transform(ts)

    # Convert to PyTorch tensor
    ts = torch.FloatTensor(ts)

    # Create sequences
    sequences = create_sequences(ts, lookback_horizon, forecast_horizon)

    # Split into training and validation sets
    train_sequences, val_sequences = train_test_split(
        sequences, test_size=0.2, random_state=42
    )

    # Create tensor datasets
    train_data = TensorDataset(
        torch.cat([x[0] for x in train_sequences]).view(
            len(train_sequences), lookback_horizon
        ),
        torch.cat([x[1] for x in train_sequences]).view(
            len(train_sequences), forecast_horizon
        ),
    )
    val_data = TensorDataset(
        torch.cat([x[0] for x in val_sequences]).view(
            len(val_sequences), lookback_horizon
        ),
        torch.cat([x[1] for x in val_sequences]).view(
            len(val_sequences), forecast_horizon
        ),
    )

    # Create data loaders
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)

    return train_dataloader, val_dataloader, scaler


# Set up hyperparameters
lookback_horizon = 120
forecast_size = 24
batch_size = 256
num_nhits_blocks = 3
input_size = lookback_horizon
mlp_layer_num = 3
hidden_size = 512
# num_freqs = 10
pooling_kernel_size = [2, 2, 2]
downsampling_ratios = [24, 12, 1]
bc_coefs_size = [
    lookback_horizon // downsampling_ratio for downsampling_ratio in downsampling_ratios
]
fc_coefs_size = [
    forecast_size // downsampling_ratio for downsampling_ratio in downsampling_ratios
]
dropout_prob = 0.0
num_samples = 1000

df = pd.read_csv("./data/ETTm2.csv")

features = df.select_dtypes(include=[np.number]).columns.tolist()


# List to store MSE for each feature
mse_list = []

# List of features in your DataFrame
df = df[["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]]

features = df.columns

for feature in features:
    # Prepare the data
    train_dataloader, val_dataloader, scaler = prepare_data(
        df, feature, lookback_horizon, forecast_size, batch_size
    )

    # Instantiate the model
    model = NHiTS(
        lookback_horizon,
        num_nhits_blocks,
        input_size,
        mlp_layer_num,
        hidden_size,
        bc_coefs_size,
        fc_coefs_size,
        pooling_kernel_size,
        forecast_size,
        pooling_mode="max",
        dropout_prob=0.0,
    )

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Training loop
    for epoch in range(5):
        for seq, labels in train_dataloader:
            # Forward pass
            outputs = model(seq)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)

        # Validation
        with torch.no_grad():
            val_loss = 0
            for seq, labels in val_dataloader:
                outputs = model(seq)
                val_loss += criterion(outputs, labels)
            val_loss /= len(val_dataloader)

        print(
            f"Epoch {epoch}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}"
        )
        mse_list.append(val_loss.item())

    # Saving the model
    torch.save(model.state_dict(), f"./model_{feature}.pth")

# Calculate average MSE over all features
avg_mse = sum(mse_list) / len(mse_list)
print(f"Average MSE over all features: {avg_mse}")

# # Prepare the data
# train_dataloader, val_dataloader, scaler = prepare_data(df, 'OT', lookback_horizon, forecast_horizon, batch_size)

# # Instantiate the model
# model = NHiTS(lookback_horizon, num_nhits_blocks, input_size, mlp_layer_num, hidden_size, bc_coefs_size, fc_coefs_size, pooling_kernel_size, pooling_mode='max', dropout_prob=0.0)

# # Define loss function and optimizer
# criterion = nn.MSELoss()
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# # Training loop
# for epoch in range(50):
#     for seq, labels in train_dataloader:
#         # Forward pass
#         outputs = model(seq)
#         loss = criterion(outputs, labels)


#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     # Validation
#     with torch.no_grad():
#         val_loss = 0
#         for seq, labels in val_dataloader:
#             outputs = model(seq)
#             val_loss += criterion(outputs, labels)
#         val_loss /= len(val_dataloader)

#     print(f'Epoch {epoch}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}')
#     # Saving the model
# torch.save(model.state_dict(), './model.pth')
