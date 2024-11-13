import numpy as np
import os

modes_train = np.array([5, 1, 2, 3, 8, 9])
modes_test = np.array([4, 5, 0, 6, 7])
n_train = np.array([20000, 20000, 5625, 20000, 20000, 20000])
n_test = np.array([200, 200, 200, 200, 200])
n_inst_train = 3
n_inst_test = 1
N = 150
M = 50
grid_limit = 75
snr = 15
n_shards = 8


def shard_dataset(data, n_shards):
    n = len(data)
    shard_size = n // n_shards
    shards = []
    for i in range(n_shards):
        start = i * shard_size
        end = start + shard_size
        if i == n_shards - 1:
            end = n
        shards.append(data[start:end])
    return shards


def generate_dataset(N, n_freq_pairs, mode, grid_limit=75, n_instances=1, snr=15):

    delta_f = 1 / N
    a1, a2 = 1, 1
    x = []
    y = []
    modes = []

    if mode == 0:

        f1 = np.random.uniform(0, 0.5 - delta_f, n_freq_pairs)
        f2 = f1 + delta_f

    elif mode == 1:

        f1 = np.random.uniform(0, 0.5 - delta_f, n_freq_pairs)
        f2 = (
            f1
            + delta_f
            + np.random.uniform(-(f1 + delta_f), 0.5 - f1 - delta_f, n_freq_pairs)
        )

    elif mode == 2:

        grid_x = np.linspace(1, grid_limit, grid_limit)
        f1 = np.random.choice(grid_x, n_freq_pairs)
        f2 = np.array(
            [np.random.choice(np.delete(grid_x, np.where(grid_x == f))) for f in f1]
        )

        f1 *= delta_f
        f2 *= delta_f

    elif mode == 3:

        f1 = np.random.uniform(0, 0.5 - delta_f, n_freq_pairs)
        f2 = []

        for f in f1:
            min_limit = np.ceil(-f / delta_f)
            max_limit = np.floor((0.5 - f) / delta_f)
            choice_set = np.arange(min_limit, max_limit + 1)
            f2.append(f + delta_f * np.random.choice(choice_set))

        f2 = np.array(f2)

    elif mode == 4:

        f1 = np.random.uniform(0, 0.5 - 0.1 * delta_f, n_freq_pairs)
        f2 = f1 + 0.1 * delta_f

    elif mode == 5:

        f1 = np.random.uniform(0, 0.5 - 0.5 * delta_f, n_freq_pairs)
        f2 = f1 + 0.5 * delta_f

    elif mode == 6:

        f1 = np.random.uniform(0, 0.5 - 2 * delta_f, n_freq_pairs)
        f2 = f1 + 2 * delta_f

    elif mode == 7:

        f1 = np.random.uniform(0, 0.5 - 3 * delta_f, n_freq_pairs)
        f2 = f1 + 3 * delta_f

    elif mode == 8:

        f1 = np.random.uniform(0, 0.5, n_freq_pairs)
        f2 = np.random.uniform(0, 0.5, n_freq_pairs)

    elif mode == 9:

        f1 = 0.25 * np.random.randn(n_freq_pairs) + 0.25

        for f_l in range(len(f1)):
            if f1[f_l] > 0.5:
                f1[f_l] = 0.5
            if f1[f_l] < 0:
                f1[f_l] = 0

        min_lim = 0 - f1
        max_lim = 0.5 - f1

        f2 = f1 + np.random.uniform(min_lim, max_lim, n_freq_pairs)

    elif mode == 10:
        f1 = 0.1 * np.ones(n_freq_pairs)
        f2 = 0.2 * np.ones(n_freq_pairs)

    f1 *= 2 * np.pi
    f2 *= 2 * np.pi

    f1_final = []
    f2_final = []

    for i in range(n_freq_pairs):
        n = np.arange(N)
        s = a1 * np.exp(1j * f1[i] * n) + a2 * np.exp(1j * f2[i] * n)
        s_norm = np.linalg.norm(np.abs(s)) ** 2
        sigma = np.sqrt(((10 ** (-snr / 10)) * s_norm) / N)

        for _ in range(n_instances):
            s_noisy = s + np.random.randn(N) * sigma * ((1 + 1j) / np.sqrt(2))
            x.append(s_noisy)
            y.append(s[M:])
            f1_final.append(f1[i])
            f2_final.append(f2[i])
            modes.append(mode)

    return (
        np.array(x),
        np.array(y),
        np.array(f1_final),
        np.array(f2_final),
        np.array(modes),
    )


x = np.zeros((np.sum(n_train) * n_inst_train, N), dtype=complex)
y = np.zeros((np.sum(n_train) * n_inst_train, N - M), dtype=complex)
f1 = np.zeros(np.sum(n_train) * n_inst_train)
f2 = np.zeros(np.sum(n_train) * n_inst_train)
mode_vals = np.zeros(np.sum(n_train) * n_inst_train)
n_curr = 0

for i in range(len(modes_train)):
    (
        x[n_curr : n_curr + n_train[i] * n_inst_train, :],
        y[n_curr : n_curr + n_train[i] * n_inst_train, :],
        f1[n_curr : n_curr + n_train[i] * n_inst_train],
        f2[n_curr : n_curr + n_train[i] * n_inst_train],
        mode_vals[n_curr : n_curr + n_train[i] * n_inst_train],
    ) = generate_dataset(
        N,
        n_train[i],
        modes_train[i],
        grid_limit=grid_limit,
        n_instances=n_inst_train,
        snr=snr,
    )
    n_curr += n_train[i] * n_inst_train

idx = np.random.permutation(x.shape[0])
x = x[idx]
y = y[idx]
f1 = f1[idx]
f2 = f2[idx]
mode_vals = mode_vals[idx]

os.mkdir(f"data/SNR_{snr}")

folders = ["x_train", "y_train", "z_train", "f1_train", "f2_train", "modes_train"]
arrs = [x[:, :M], y, x[:, M:], f1, f2, mode_vals]

for f in range(len(folders)):
    os.mkdir(f"data/SNR_{snr}/{folders[f]}")
    shards = shard_dataset(arrs[f], n_shards)

    for i in range(n_shards):
        np.savez_compressed(
            f"data/SNR_{snr}/{folders[f]}/{folders[f]}_{i}.npz", np.array(shards[i])
        )

x = np.zeros((np.sum(n_test) * n_inst_test, N), dtype=complex)
y = np.zeros((np.sum(n_test) * n_inst_test, N - M), dtype=complex)
f1 = np.zeros(np.sum(n_test) * n_inst_test)
f2 = np.zeros(np.sum(n_test) * n_inst_test)
mode_vals = np.zeros(np.sum(n_test) * n_inst_test)
n_curr = 0

for i in range(len(modes_test)):
    (
        x[n_curr : n_curr + n_test[i] * n_inst_test, :],
        y[n_curr : n_curr + n_test[i] * n_inst_test, :],
        f1[n_curr : n_curr + n_test[i] * n_inst_test],
        f2[n_curr : n_curr + n_test[i] * n_inst_test],
        mode_vals[n_curr : n_curr + n_test[i] * n_inst_test],
    ) = generate_dataset(
        N,
        n_test[i],
        modes_test[i],
        grid_limit=grid_limit,
        n_instances=n_inst_test,
        snr=snr,
    )
    n_curr += n_test[i] * n_inst_test


idx = np.random.permutation(x.shape[0])
x = x[idx]
y = y[idx]
f1 = f1[idx]
f2 = f2[idx]
mode_vals = mode_vals[idx]

folders = ["x_test", "y_test", "z_test", "f1_test", "f2_test", "modes_test"]
arrs = [x[:, :M], y, x[:, M:], f1, f2, mode_vals]

for f in range(len(folders)):
    os.mkdir(f"data/SNR_{snr}/{folders[f]}")
    shards = shard_dataset(arrs[f], n_shards)

    for i in range(n_shards):
        np.savez_compressed(
            f"data/SNR_{snr}/{folders[f]}/{folders[f]}_{i}.npz", np.array(shards[i])
        )
