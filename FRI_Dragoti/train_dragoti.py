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

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
batch_size = 1024
n_epochs = 50
snr = 15

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
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, n_epochs, eta_min=0, last_epoch=-1
)

file_path = "train_15dB.h5"


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


with h5py.File(file_path, "r") as f:

    s_m_a_real1 = f["/s_m_a_real"][:]
    s_m_a_imag1 = f["/s_m_a_imag"][:]
    s_m_noisy_real1 = f["/s_m_noisy_real"][:]
    s_m_noisy_imag1 = f["/s_m_noisy_imag"][:]

scale_factor = np.max(s_m_a_real1)

x_train = np.empty((s_m_noisy_real1.shape[0], s_m_noisy_real1.shape[1], 2))
y_train = np.empty((s_m_a_real1.shape[0], 39, 2))

x_train[:, :, 0], x_train[:, :, 1] = s_m_noisy_real1, s_m_noisy_imag1
y_train[:, :, 0], y_train[:, :, 1] = s_m_a_real1[:, 21:], s_m_a_imag1[:, 21:]

x_train = x_train / scale_factor
y_train = y_train / scale_factor

x_train = torch.tensor(x_train).float()
y_train = torch.tensor(y_train).float()

del s_m_noisy_real1, s_m_noisy_imag1, s_m_a_real1, s_m_a_imag1

f1 = "partition1/"
files = os.listdir(f1)
id_p1 = []
for i in files:
    id_p1.append(np.load(f1 + i))

id_p1 = np.concatenate(id_p1, axis=0)

f1 = "partition2/"
files = os.listdir(f1)
id_p2 = []
for i in files:
    id_p2.append(np.load(f1 + i))

id_p2 = np.concatenate(id_p2, axis=0)
x_train1 = x_train[id_p1]
y_train1 = y_train[id_p1]
x_train2 = x_train[id_p2]
y_train2 = y_train[id_p2]

train_dataset = torch.utils.data.TensorDataset(x_train1, y_train1)
train_loader1 = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

train_dataset = torch.utils.data.TensorDataset(x_train2, y_train2)
train_loader2 = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

del x_train, y_train, train_dataset


def test():
    model.eval()
    savepath = "test/deltat1_15dB1_60.h5"
    # savepath = 'test_new/deltat1_15dBhisam_new1_60.h5'

    with h5py.File(savepath, "r") as f:

        s_m_real_test1 = f["/s_m_noisy_real"][:]
        s_m_imag_test1 = f["/s_m_noisy_imag"][:]

        scale_factor = np.max(f["/s_m_a_real"][:])

    x_test = np.empty((s_m_real_test1.shape[0], s_m_real_test1.shape[1], 2))

    x_test[:, :, 0], x_test[:, :, 1] = s_m_real_test1, s_m_imag_test1

    x_test = x_test / scale_factor

    del s_m_real_test1, s_m_imag_test1

    predicted_output = (
        model(torch.tensor(x_test).float().to(device)).cpu().detach().numpy()
    )

    err_value_1 = []
    a = 0.1
    b = 0.1 + (1 / 60)
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


def train(model, tl1, tl2, optimizer, loss_fn, device):
    model.train()
    losses = []

    for epoch in range(n_epochs):
        total_loss = 0
        for batch, (src, tgt) in enumerate(tl1):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src)
            loss = loss_fn(output, tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        for batch, (src, tgt) in enumerate(tl2):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src)
            loss = loss_fn(output, tgt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        losses.append(np.round(total_loss / (len(tl1) + len(tl2)), 4))
        print(
            f"Epoch {epoch + 1}/{n_epochs} : Loss --> {np.round(total_loss / (len(tl1) + len(tl2)), 4)}"
        )
        test()
        print(scheduler.get_last_lr())
        print("--------------------------------------------------")
        scheduler.step()

    return np.array(losses)


train_loss = train(model, train_loader2, train_loader1, optimizer, loss_fn, device)

model.eval()
torch.save(model.state_dict(), f"model_{n_epochs}_{snr}dB.pt")

print("Training completed")
