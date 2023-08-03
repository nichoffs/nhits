import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from fourier_forecaster import FourierForecaster

# Read the wav file
rate, data = wavfile.read("./data/violin.wav")

# Normalize data
data = data / data.max()

# Convert to tensor and reshape to (B, num_samples)
data = torch.tensor(data).float().view(1, -1)

# Create time tensor
t = torch.linspace(0, len(data[0]) / rate, len(data[0])).unsqueeze(0)
y = data

# plt.figure(figsize=(15,5))
# plt.plot(t[0].numpy(), data[0].numpy())
# plt.title("Original Violin Sound")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.show()

# Number of sine wave components
num_freqs = 10

# Initialize the amplitudes, frequencies, and phases
amps = nn.Parameter(torch.randn((1, num_freqs, 1)).float(), requires_grad=True)
freqs = nn.Parameter(torch.randn((1, num_freqs, 1)).float(), requires_grad=True)
phases = nn.Parameter(torch.zeros((1, num_freqs, 1)).float(), requires_grad=True)

# Initialize the Fourier forecaster
forecaster = FourierForecaster(num_freqs)

# Initialize the optimizer
optim = torch.optim.AdamW([amps, freqs, phases], lr=3e-4)

losses = torch.tensor([])

# Training loop
for i in range(6000):
    # Forward pass: compute predicted y
    y_pred = forecaster(t, amps, freqs, phases)

    # Compute and record loss
    loss = nn.L1Loss()(y_pred, y)
    losses = torch.cat((losses, torch.tensor([loss])))

    # Zero gradients, perform a backward pass, and update the weights
    optim.zero_grad()
    loss.backward()
    optim.step()

    if i % 100 == 0:
        print(f"epoch {i}: loss {sum(losses[-500:]/500)}")

# Plot the original and the predicted signals
plt.figure(figsize=(15, 5))
plt.plot(t[0].numpy(), data[0].numpy(), label="Original")
plt.plot(t[0].numpy(), y_pred[0].detach().numpy(), label="Predicted")
plt.title("Original and Predicted Violin Sound")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()
