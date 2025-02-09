import numpy as np
import os
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from model import Transformer_Model
from esprit import ESPRIT

import argparse

def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(model, train_loader, optimizer, loss_fn, n_epochs, device):
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

def get_args():
    """Parse command-line arguments for inference."""
    parser = argparse.ArgumentParser(description="Parse transformer model parameters.")
    parser.add_argument("--d_model", type=int, default=64, help="Dimension of model")
    parser.add_argument("--ff", type=int, default=256, help="Feedforward dimension")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--n_head", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--device", type=str, default="cuda", help="Device to inference on")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--snr", type=int, default=15, help="Signal-to-noise ratio")
    parser.add_argument("--M", type=int, default=50, help="Number of samples")
    parser.add_argument("--out_length", type=int, default=100, help="Output length")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    
    return parser.parse_args()

def print_arguments(args):
    print("Inference with the following model parameters:")
    print(f"d_model: {args.d_model}")
    print(f"ff: {args.ff}")
    print(f"n_layers: {args.n_layers}")
    print(f"dropout: {args.dropout}")
    print(f"n_head: {args.n_head}")
    print(f"SNR: {args.snr}")
    print(f"M: {args.M}")
    print(f"out_length: {args.out_length}")
    print(f"n_epochs: {args.n_epochs}")
    print(f"batch_size: {args.batch_size}")
    print(f"device: {args.device}")
    print(f"seed: {args.seed}")

if __name__ == "__main__":
    args = get_args()
    print_arguments(args)

    seed_value = args.seed
    set_seed(seed_value)

    device = args.device
    batch_first = True

    model = Transformer_Model(
        args.d_model,
        args.n_head,
        args.ff,
        args.n_layers,
        args.dropout,
        args.M,
        args.out_length,
        batch_first,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    path = f"data/SNR_{args.snr}/"
    folders = os.listdir(path)

    x_c = np.load(path + "x_train.npz")["arr_0"]
    y_c = np.load(path + "y_train.npz")["arr_0"]

    x_train = np.empty((x_c.shape[0], x_c.shape[1], 2))
    y_train = np.empty((y_c.shape[0], y_c.shape[1], 2))

    x_train[:, :, 0], x_train[:, :, 1] = np.real(x_c), np.imag(x_c)
    y_train[:, :, 0], y_train[:, :, 1] = np.real(y_c), np.imag(y_c)

    x_train = torch.tensor(x_train).float()
    y_train = torch.tensor(y_train).float()

    del x_c, y_c

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    train_loss = train(model, train_loader, optimizer, loss_fn, args.n_epochs, device)

    model_path = f"models/model_dim{args.d_model}_ff{args.ff}_layers{args.n_layers}_dropout{args.dropout}_nhead_{args.n_head}_SNR_{args.snr}.pt"
    torch.save(model.state_dict(), model_path)

    del x_train, y_train, train_dataset, train_loader
