import numpy as np


def get_base(size, fs=250, n_components=10, low=7, high=14):
    t = np.arange(size) / fs
    base = []
    freqs = np.linspace(low, high, n_components)
    for model_freq in freqs:
        base += [np.sin(model_freq * t * 2 * np.pi), np.cos(model_freq * t * 2 * np.pi)]
    base = np.vstack(base).T
    return base, freqs
