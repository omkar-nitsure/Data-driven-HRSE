import os
import numpy as np
import torch
import util
import matplotlib.pyplot as plt
from data import fr

snr = 30
fr_path = f"models/SNR_{snr}/fr/epoch_100.pth"
fc_path = f"models/SNR_{snr}/fc/epoch_100.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fr_module, _, _, _, _ = util.load(fr_path, "fr", device)
fr_module.cpu()
fr_module.eval()

fc_module, _, _, _, _ = util.load(fc_path, "fc", device)
fc_module.cpu()
fc_module.eval()

xgrid = np.linspace(0, 0.5, fr_module.fr_size, endpoint=False)

path = f"../data/SNR_{snr}/"

s_temp = np.load(path + "x_test.npz")["arr_0"]
f1 = np.load(path + "f1_test.npz")["arr_0"]
f2 = np.load(path + "f2_test.npz")["arr_0"]
modes_test = np.load(path + "modes_test.npz")["arr_0"]

nfreq = 2 * np.ones(s_temp.shape[0], dtype="int")

f = np.zeros((f1.shape[0], 10))
f[:, 0] = f1
f[:, 1] = f2
f = f / (2 * np.pi)

s = np.zeros((s_temp.shape[0], 2, 50))
s[:, 0, :] = np.real(s_temp)
s[:, 1, :] = np.imag(s_temp)

del f1, f2, s_temp

s = s.astype("float32")
f = f.astype("float32")

f_est = []

for idx in range(len(s)):
    with torch.no_grad():
        fr_15dB = fr_module(torch.tensor(s[idx][None]))
        nestimate_15dB = fc_module(fr_15dB).numpy().round()

    fr_15dB = fr_15dB.numpy()
    f_est.append(fr.find_freq(fr_15dB, nestimate_15dB, xgrid))

f_est = np.array(f_est)
f_est = f_est.reshape(f_est.shape[0], f_est.shape[2])

f_pred = []
f_true = []

for i in range(len(f_est)):
    f1 = []
    f2 = []
    for j in range(len(f_est[i])):
        if f_est[i][j] != -1:
            f1.append(f_est[i][j])
        if f[i][j] != 0:
            f2.append(f[i][j])

    f_pred.append(np.sort(np.array(f1)))
    f_true.append(np.sort(np.array(f2)))


err = np.zeros(5)
counts = np.zeros(5)

modes = {4: 0, 5: 1, 0: 2, 6: 3, 7: 4}

for i in range(len(f_pred)):

    num = 0.0
    den = 0.0

    for j in range(len(f_pred[i])):
        num += (f_pred[i][j] - f_true[i][j]) ** 2
        den += f_true[i][j] ** 2

    err[modes[modes_test[i]]] += 10 * np.log10(num / den)
    counts[modes[modes_test[i]]] += 1

print(err / counts)
