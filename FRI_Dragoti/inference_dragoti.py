import numpy as np
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformer_model import Transformer_Model
from esprit import ESPRIT

import tables
import h5py

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"device: {device}")

d_model = 64
context_length = 21
n_head = 8
ff = 1024
n_layers = 2
dropout = 0.0
out_length = 39
batch_first = True
snr = 15
n_epochs = 50
model_path = f"model_{n_epochs}_{snr}dB.pt"

model = Transformer_Model(
    d_model,
    context_length,
    n_head,
    ff,
    n_layers,
    dropout,
    out_length,
    batch_first,
).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()


def mse_db1(j, k, a, b, f1):
    z1 = (j - a) ** 2
    z2 = (k - b) ** 2
    sum1 = (abs(z1) + abs(z2)) / f1
    sum2 = ((a**2) + (b**2)) / f1
    if sum2 == 0 or np.isinf(sum2):
        err = -1
    else:

        if sum1 == 0:
            err = -1
        else:
            err = 10 * np.log10(sum1 / sum2)

    return err


# path = 'test_new/deltat1_15dBhisam_new5_60.h5'
path = "test/deltat1_15dB1_2.h5"

with h5py.File(path, "r") as f:

    s_m_real_test1 = f["/s_m_noisy_real"][:]
    s_m_imag_test1 = f["/s_m_noisy_imag"][:]

    scale_factor = np.max(f["/s_m_a_real"][:])


x_test = np.empty((s_m_real_test1.shape[0], s_m_real_test1.shape[1], 2))

x_test[:, :, 0], x_test[:, :, 1] = s_m_real_test1, s_m_imag_test1
x_test = x_test / scale_factor

del s_m_real_test1, s_m_imag_test1

predicted_output = model(torch.tensor(x_test).float().to(device)).cpu().detach().numpy()

num_simulations = 1000
err_value_1 = []
err_value_2 = []
err_value_3 = []
a = 0.1
b = 0.1 + (0.5 / 60)
f1 = 2

x_ = np.concatenate((x_test, predicted_output), axis=1)
x_esp = x_[:, :, 0] + 1j * x_[:, :, 1]

del x_test, predicted_output, x_

for i in range(len(x_esp)):
    x = x_esp[i]
    k1 = ESPRIT(x, 2, 30) / (2 * np.pi)
    mse_1 = mse_db1(k1[0], k1[1], a, b, f1)
    err_value_1.append(mse_1)

avg_err = np.mean(err_value_1)
print(f"Average error: {avg_err}")
