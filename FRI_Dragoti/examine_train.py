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


file_path = "train_15dB.h5"

with h5py.File(file_path, "r") as f:

    s_m_a_real1 = f["/s_m_a_real"][:]
    s_m_a_imag1 = f["/s_m_a_imag"][:]

x_train = np.empty((s_m_a_real1.shape[0], s_m_a_real1.shape[1], 2))
x_train[:, :, 0], x_train[:, :, 1] = s_m_a_real1, s_m_a_imag1

del s_m_a_real1, s_m_a_imag1

f1 = 2
x_esp = x_train[:, :, 0] + 1j * x_train[:, :, 1]

del x_train

k0 = []
k1 = []
p1 = 10 * np.array([300, 600, 600, 600, 1200, 1200, 1200, 1200, 5000])
p2 = 10 * 300 * np.ones(9)


for i in range(len(x_esp)):
    x = x_esp[i]

    k = ESPRIT(x, 2, 30) / (2 * np.pi)
    k0.append(k[0])
    k1.append(k[1])

k0 = np.array(k0)
k1 = np.array(k1)
res = k1 - k0
scale = 1 / 60
idx = {}
ideal_res = scale * np.array([0.1, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5])

occurances = np.zeros(9)

ids = {}
for i in range(ideal_res.shape[0]):
    ids[i] = []

for i in range(len(res)):
    occurances[np.argmin(np.abs(res[i] - ideal_res))] += 1
    ids[np.argmin(np.abs(res[i] - ideal_res))].append(i)

ids1 = {}
ids2 = {}

for i in range(len(list(ids.keys()))):
    ids[i] = np.array(ids[i])
for i in range(len(list(ids.keys()))):
    ids1[i] = list(ids[i][np.random.randint(0, len(ids[i]), int(p1[i]))])
    ids2[i] = list(ids[i][np.random.randint(0, len(ids[i]), int(p2[i]))])

import os

for i in range(len(list(ids1.keys()))):
    np.save(f"partition1/{i}.npy", ids1[i])
for i in range(len(list(ids2.keys()))):
    np.save(f"partition2/{i}.npy", ids2[i])


print(f"Average error: {100*(occurances/np.sum(occurances))}")
