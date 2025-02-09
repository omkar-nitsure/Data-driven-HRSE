import numpy as np
import torch
import os

def frequency_generator(f, nf, min_sep, dist_distribution):
    if dist_distribution == 'random':
        random_freq(f, nf, min_sep)
    elif dist_distribution == 'jittered':
        jittered_freq(f, nf, min_sep)
    elif dist_distribution == 'normal':
        normal_freq(f, nf, min_sep)


def random_freq(f, nf, min_sep):
    """
    Generate frequencies uniformly.
    """
    for i in range(nf):
        f_new = np.random.rand() - 1 / 2
        condition = True
        while condition:
            f_new = np.random.rand() - 1 / 2
            condition = (np.min(np.abs(f - f_new)) < min_sep) or \
                        (np.min(np.abs((f - 1) - f_new)) < min_sep) or \
                        (np.min(np.abs((f + 1) - f_new)) < min_sep)
        f[i] = f_new


def jittered_freq(f, nf, min_sep, jit=1):
    """
    Generate jittered frequencies.
    """
    l, r = -0.5, 0.5 - nf * min_sep * (1 + jit)
    s = l + np.random.rand() * (r - l)
    c = np.cumsum(min_sep * (np.ones(nf) + np.random.rand(nf) * jit))
    f[:nf] = (s + c - min_sep + 0.5) % 1 - 0.5


def normal_freq(f, nf, min_sep, scale=0.05):
    """
    Distance between two frequencies follows a normal distribution
    """
    f[0] = np.random.uniform() - 0.5
    for i in range(1, nf):
        condition = True
        while condition:
            d = np.random.normal(scale=scale)
            f_new = (d + np.sign(d) * min_sep + f[i - 1] + 0.5) % 1 - 0.5
            condition = (np.min(np.abs(f - f_new)) < min_sep) or \
                        (np.min(np.abs((f - 1) - f_new)) < min_sep) or \
                        (np.min(np.abs((f + 1) - f_new)) < min_sep)
        f[i] = f_new


def amplitude_generation(dim, amplitude, floor_amplitude=0.1):
    """
    Generate the amplitude associated with each frequency.
    """
    if amplitude == 'uniform':
        return np.random.rand(*dim) * (1 - floor_amplitude) + floor_amplitude
    elif amplitude == 'normal':
        return np.abs(np.random.randn(*dim))
    elif amplitude == 'normal_floor':
        return np.abs(np.random.randn(*dim)) + floor_amplitude
    elif amplitude == 'alternating':
        return np.random.rand(*dim) * 0.5 + 20 * np.random.rand(*dim) * np.random.randint(0, 2, size=dim)

def gen_signal(num_samples, signal_dim, num_freq, min_sep, distance='normal', amplitude='normal_floor',
               floor_amplitude=0.1, variable_num_freq=False):
    
    nfreq = 2*np.ones(num_samples, dtype='int')

    path = "data/SNR_30/"

    s_temp = np.load(path + 'x_train.npz')['arr_0']
    f1 = np.load(path + 'f1_train.npz')['arr_0']
    f2 = np.load(path + 'f2_train.npz')['arr_0']

    f = np.zeros((f1.shape[0], 2))
    f[:, 0] = f1
    f[:, 1] = f2
    f = f / (2*np.pi)

    s = np.zeros((s_temp.shape[0], 2, signal_dim))
    s[:, 0, :] = np.real(s_temp)
    s[:, 1, :] = np.imag(s_temp)

    return s.astype('float32'), f.astype('float32'), nfreq

def compute_snr(clean, noisy):
    return np.linalg.norm(clean, axis=1) ** 2 / np.linalg.norm(clean - noisy, axis=1) ** 2


def compute_snr_torch(clean, noisy):
    return (torch.sum(clean.view(clean.size(0), -1) ** 2, dim=1) / torch.sum(
        ((clean - noisy).view(clean.size(0), -1)) ** 2, dim=1)).mean()
