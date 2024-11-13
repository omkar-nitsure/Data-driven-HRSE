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
model_path = f"models/model_SNR_{snr}.pt"

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
model.load_state_dict(torch.load(model_path))
model.eval()

loss_fn = nn.MSELoss()

path = f"data/SNR_{snr}/"
folders = os.listdir(path)

xt_c = []
yt_c = []
zt_c = []
f1t = []
f2t = []
modes = []

for folder in folders:
    for file in os.listdir(path + folder):
        if "x_test" in file:
            xt_c.append(np.load(path + folder + "/" + file)["arr_0"])
        elif "y_test" in file:
            yt_c.append(np.load(path + folder + "/" + file)["arr_0"])
        elif "z_test" in file:
            zt_c.append(np.load(path + folder + "/" + file)["arr_0"])
        elif "f1_test" in file:
            f1t.append(np.load(path + folder + "/" + file)["arr_0"])
        elif "f2_test" in file:
            f2t.append(np.load(path + folder + "/" + file)["arr_0"])
        elif "modes_test" in file:
            modes.append(np.load(path + folder + "/" + file)["arr_0"])

xt_c = np.concatenate(xt_c, axis=0)
yt_c = np.concatenate(yt_c, axis=0)
zt_c = np.concatenate(zt_c, axis=0)
f1 = np.concatenate(f1t, axis=0)
f2 = np.concatenate(f2t, axis=0)
modes = np.concatenate(modes, axis=0)


x_test = np.empty((xt_c.shape[0], xt_c.shape[1], 2))
y_test = np.empty((yt_c.shape[0], yt_c.shape[1], 2))
z_test = np.empty((zt_c.shape[0], zt_c.shape[1], 2))

x_test[:, :, 0], x_test[:, :, 1] = np.real(xt_c), np.imag(xt_c)
y_test[:, :, 0], y_test[:, :, 1] = np.real(yt_c), np.imag(yt_c)
z_test[:, :, 0], z_test[:, :, 1] = np.real(zt_c), np.imag(zt_c)

x_test = torch.tensor(x_test).float()
y_test = torch.tensor(y_test).float()
z_test = torch.tensor(z_test).float()

del xt_c, yt_c, zt_c, f1t, f2t

def test(model, x_test, y_test, device):
    model.eval()
    with torch.no_grad():
        x_test, y_test = x_test.to(device), y_test.to(device)
        output = model(x_test)
        loss = loss_fn(output, y_test)
        print(f"Loss: {loss.item()}")
        return output


output = test(model, x_test, y_test, device)
output = output.cpu().detach().numpy()


xt1 = (
    x_test[:, :, 0].cpu().detach().numpy() + 1j * x_test[:, :, 1].cpu().detach().numpy()
)

xt2 = np.empty((xt1.shape[0], M + out_length), dtype=complex)
xt2[:, :M] = xt1
xt2[:, M:] = (
    z_test[:, :, 0].cpu().detach().numpy() + 1j * z_test[:, :, 1].cpu().detach().numpy()
)

xt3 = np.empty((xt1.shape[0], M + out_length), dtype=complex)
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

print(f"xt1: {err1/counts}")

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

print(f"xt2: {err2/counts}")

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

print(f"xt3: {err3/counts}")

x_axis = ["0.1/150", "0.5/150", "1/150", "2/150", "3/150"]
plt.figure(figsize=(10, 6))
plt.plot(x_axis, err1 / counts, label="METHOD-1: 50(T)", color="blue")
plt.plot(x_axis, err2 / counts, label="METHOD-2: 150(T)", color="red")
plt.plot(x_axis, err3 / counts, label="METHOD-3: 50(T) & 100(P)", color="orange")
plt.xticks(x_axis)
plt.xlabel("RESOLUTION")
plt.ylabel("MSE in dB")
plt.legend()
plt.grid()
plt.savefig(f"plots/NMSE_SNR_{snr}.png")

np.save(
    f"plotting_data/errs_SNR_{snr}.npy",
    np.vstack((err1 / counts, err2 / counts, err3 / counts)),
)
