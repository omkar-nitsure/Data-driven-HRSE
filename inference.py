import numpy as np
import os
import matplotlib.pyplot as plt
import random
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

def test(model, x_test, y_test, device):
    model.eval()
    with torch.no_grad():
        x_test, y_test = x_test.to(device), y_test.to(device)
        output = model(x_test)
        loss = loss_fn(output, y_test)
        print(f"Loss: {loss.item()}")
        return output

def get_args():
    """Parse command-line arguments for inference."""
    parser = argparse.ArgumentParser(description="Parse transformer model parameters.")
    parser.add_argument("--d_model", type=int, default=64, help="Dimension of model")
    parser.add_argument("--ff", type=int, default=256, help="Feedforward dimension")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--n_head", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--device", type=str, default="cpu", help="Device to inference on")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--snr", type=int, default=15, help="Signal-to-noise ratio")
    parser.add_argument("--M", type=int, default=50, help="Number of samples")
    parser.add_argument("--out_length", type=int, default=100, help="Output length")
    
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
    print(f"device: {args.device}")
    print(f"seed: {args.seed}")

if __name__ == "__main__":
    args = get_args()
    print_arguments(args)

    seed_value = args.seed
    set_seed(seed_value)
    device = args.device
    batch_first = True

    model_path = f"models/model_dim{args.d_model}_ff{args.ff}_layers{args.n_layers}_dropout{args.dropout}_nhead_{args.n_head}_SNR_{args.snr}.pt"

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
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")

    loss_fn = nn.MSELoss()

    path = f"data/SNR_{args.snr}/"

    xt_c = np.load(path + "x_test.npz")["arr_0"]
    yt_c = np.load(path + "y_test.npz")["arr_0"]
    zt_c = np.load(path + "z_test.npz")["arr_0"]
    f1 = np.load(path + "f1_test.npz")["arr_0"]
    f2 = np.load(path + "f2_test.npz")["arr_0"]
    modes = np.load(path + "modes_test.npz")["arr_0"]

    x_test = np.empty((xt_c.shape[0], xt_c.shape[1], 2))
    y_test = np.empty((yt_c.shape[0], yt_c.shape[1], 2))
    z_test = np.empty((zt_c.shape[0], zt_c.shape[1], 2))

    x_test[:, :, 0], x_test[:, :, 1] = np.real(xt_c), np.imag(xt_c)
    y_test[:, :, 0], y_test[:, :, 1] = np.real(yt_c), np.imag(yt_c)
    z_test[:, :, 0], z_test[:, :, 1] = np.real(zt_c), np.imag(zt_c)

    x_test = torch.tensor(x_test).float()
    y_test = torch.tensor(y_test).float()
    z_test = torch.tensor(z_test).float()

    del xt_c, yt_c, zt_c

    output = test(model, x_test, y_test, device)
    output = output.cpu().detach().numpy()


    xt1 = (
        x_test[:, :, 0].cpu().detach().numpy() + 1j * x_test[:, :, 1].cpu().detach().numpy()
    )

    xt2 = np.empty((xt1.shape[0], args.M + args.out_length), dtype=complex)
    xt2[:, :args.M] = xt1
    xt2[:, args.M:] = (
        z_test[:, :, 0].cpu().detach().numpy() + 1j * z_test[:, :, 1].cpu().detach().numpy()
    )

    xt3 = np.empty((xt1.shape[0], args.M + args.out_length), dtype=complex)
    xt3[:, : xt1.shape[1]] = xt1
    xt3[:, xt1.shape[1] :] = output[:, :, 0] + 1j * output[:, :, 1]

    err1 = np.zeros(5)
    counts = np.zeros(5)

    idx = {
        0: 2,
        4: 0,
        5: 1,
        6: 3,
        7: 4,
    }

    for i in range(len(xt1)):
        w = ESPRIT(xt1[i], 2, 25)
        freqs = np.sort(np.array([f1[i], f2[i]]))

        num = 0.0
        den = 0.0
        for j in range(len(w)):
            num += (w[j] - freqs[j]) ** 2
            den += freqs[j] ** 2

        err1[idx[int(modes[i])]] += 10 * np.log10(num / den)
        counts[idx[int(modes[i])]] += 1

    print(f"{args.M}T: {err1/counts}")

    err2 = np.zeros(5)

    for i in range(len(xt2)):
        w = ESPRIT(xt2[i], 2, 75)
        freqs = np.sort(np.array([f1[i], f2[i]]))

        num = 0.0
        den = 0.0

        for j in range(len(w)):
            num += (w[j] - freqs[j]) ** 2
            den += freqs[j] ** 2

        err2[idx[int(modes[i])]] += 10 * np.log10(num / den)

    print(f"{args.M + args.out_length}T: {err2/counts}")

    err3 = np.zeros(5)

    for i in range(len(xt3)):
        w = ESPRIT(xt3[i], 2, 75)
        freqs = np.sort(np.array([f1[i], f2[i]]))

        num = 0.0
        den = 0.0

        for j in range(len(w)):
            num += (w[j] - freqs[j]) ** 2
            den += freqs[j] ** 2

        err3[idx[int(modes[i])]] += 10 * np.log10(num / den)

    print(f"{args.M}T + {args.out_length}P: {err3/counts}")
