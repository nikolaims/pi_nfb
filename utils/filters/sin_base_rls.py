import numpy as np
from itertools import combinations
from utils.sinbase import get_base


def sin_base_rls_filter(x, fs, main_freq, n_components=16):
    n = len(x)
    ww = 0.9
    base, freqs = get_base(n, fs=fs, n_components=n_components, low=main_freq - ww, high=main_freq + ww, nonlinear=1.25)
    # prepare sin
    projections = []
    pairs = lambda: combinations(range(2 * n_components), 2)
    for i, j in pairs():
        freq = (1 - 2 * (i % 2)) * np.pi / 2 * ((i - j) % 2) + (freqs[j // 2] - freqs[i // 2]) * 2 * np.pi * np.arange(
            n) / fs
        projections.append(2 * np.cos(freq))
    projections = np.array(projections + [np.ones((n,)) for k in range(2 * n_components)])
    c_ind = np.array(list(pairs()) + [(k, k) for k in range(2 * n_components)]).T
    # rls
    from utils.models.rls import DelayedRLSPredictor
    rls = DelayedRLSPredictor(2 * n_components, mu=1, M=1, delay=0, lambda_=0.9991)
    rls_signal = np.zeros_like(x)
    rls_envelope = np.zeros_like(x)
    ws = np.zeros((n, 2 * n_components))
    for k in range(n):
        rls_signal[k] = rls.predict(base[k], x[k])
        w = rls._w
        ws[k] = w
        rls_envelope[k] = (np.sum(projections[:, k] * w[c_ind[0]] * w[c_ind[1]])) ** 0.5
    return rls_signal, rls_envelope