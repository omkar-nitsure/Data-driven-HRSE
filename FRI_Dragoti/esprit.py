import numpy as np


def ESPRIT(y, n, m):
    """
    Perform the ESPRIT (Estimation of Signal Parameters via Rotational Invariance Techniques) algorithm.

    Parameters:
    y (array-like): Input signal, should be a 1D array or list.
    n (int): Number of signal sources.
    m (int): Number of snapshots (length of the sliding window).

    Returns:
    numpy.ndarray: Estimated angles (in radians) of the signal sources.
    """

    y = np.asarray(y).flatten()
    N = len(y)

    R = np.zeros((m, m), dtype=complex)

    for i in range(m, N):
        segment = y[i : i - m : -1]
        R += np.outer(segment, segment.conj()) / N

    U, _, _ = np.linalg.svd(R)

    S = U[:, :n]
    phi = np.linalg.lstsq(S[:-1, :], S[1:, :], rcond=None)[0]

    eigenvalues = np.linalg.eigvals(phi)

    w = -np.angle(eigenvalues)

    w[w < 0] = 2 * np.pi + w[w < 0]

    return np.sort(w)
