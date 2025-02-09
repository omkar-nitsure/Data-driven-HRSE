import numpy as np


def ESPRIT(y, n, m):

    y = np.asarray(y).flatten()
    N = len(y)

    R = np.zeros((m, m), dtype=complex)

    for i in range(m, N):
        segment = y[i : i - m : -1]
        R += np.outer(segment, segment.conj()) / N

    U, _, _ = np.linalg.svd(R)

    S = U[:, :n]

    phi = np.linalg.pinv(S[:-1, :]) @ S[1:, :]

    eigenvalues = np.linalg.eigvals(phi)

    w = -np.angle(eigenvalues)

    w[w < 0] = 2 * np.pi + w[w < 0]

    return np.sort(w)
