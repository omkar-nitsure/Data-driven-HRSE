import numpy as np
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformer_model import Transformer_Model
from esprit import ESPRIT

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"device: {device}")

d_model = 64
context_length = 50
n_head = 8
ff = 1024
n_layers = 2
dropout = 0.0
leaky_slope = 0.2
M = 50
out_length = 100
batch_first = True
batch_size = 2048
n_epochs = 50
snr = 30

model = Transformer_Model(
    d_model,
    context_length,
    n_head,
    ff,
    n_layers,
    dropout,
    leaky_slope,
    M,
    out_length,
    batch_first,
).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

path = f"data/SNR_{snr}/"
folders = os.listdir(path)

x_c = []
y_c = []

for folder in folders:
    for file in os.listdir(path + folder):
        if "x_train" in file:
            x_c.append(np.load(path + folder + "/" + file)["arr_0"])
        elif "y_train" in file:
            y_c.append(np.load(path + folder + "/" + file)["arr_0"])


x_c = np.concatenate(x_c, axis=0)
y_c = np.concatenate(y_c, axis=0)

x_train = np.empty((x_c.shape[0], x_c.shape[1], 2))
y_train = np.empty((y_c.shape[0], y_c.shape[1], 2))

x_train[:, :, 0], x_train[:, :, 1] = np.real(x_c), np.imag(x_c)
y_train[:, :, 0], y_train[:, :, 1] = np.real(y_c), np.imag(y_c)

x_train = torch.tensor(x_train).float()
y_train = torch.tensor(y_train).float()

del x_c, y_c

train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)


def train(model, train_loader, optimizer, loss_fn, device):
    model.train()
    losses = []

    for epoch in range(n_epochs):
        total_loss = 0
        for batch, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src)
            loss = loss_fn(output, tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        losses.append(np.round(total_loss / len(train_loader), 4))
        print(f"Epoch {epoch + 1}: Loss: {np.round(total_loss / len(train_loader), 4)}")

    return np.array(losses)


train_loss = train(model, train_loader, optimizer, loss_fn, device)

plt.plot(np.arange(1, n_epochs + 1), train_loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
plt.savefig("plots/training_loss.png")

torch.save(model.state_dict(), f"models/model_SNR_{snr}.pt")

del x_train, y_train, train_dataset, train_loader
