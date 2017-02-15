import numpy as np


def get_base(size, fs=250, n_components=10, low=7, high=14, nonlinear=1.1, show_freqs=False):
    t = np.arange(size) / fs
    base = []
    if nonlinear>0:
        n = np.linspace(0, 1, n_components//2+1) ** nonlinear
        n = np.hstack((-n[::-1], n[1:]))[1:]
        freqs = n * (high - low) / 2 + (high +low) / 2
    else:
        freqs = np.linspace(low, high, n_components, )

    if show_freqs:
        import pylab as plt
        plt.plot(freqs, np.ones_like(freqs), 'o')
        plt.show()
        print(freqs)
    for model_freq in freqs:
        base += [np.sin(model_freq * t * 2 * np.pi), np.cos(model_freq * t * 2 * np.pi)]
    base = np.vstack(base).T
    return base, freqs
