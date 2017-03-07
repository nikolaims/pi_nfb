import numpy as np

import pylab as plt

class RCEnvelopeDetector:
    def __init__(self, factor):
        self.factor = factor
        self.y = 0

    def get_envelope(self, x):
        x = abs(x)
        self.y = x if x > self.y else self.y * self.factor + x * (1 - self.factor)
        return self.y

class RCEnvelopeDetector:
    def __init__(self, factor):
        self.factor = factor
        self.y = 0

    def get_envelope(self, x):
        x = abs(x)
        self.y = x if x > self.y else self.y * self.factor + x * (1 - self.factor)
        return self.y


if __name__ == '__main__':
    from utils.data.loaders import load_feedback
    from utils.filters import dc_blocker

    data, _signal, _derived = load_feedback(ica_artifact=True)
    data = dc_blocker(data)
    fs = 250
    nq = 250 / 2

    from scipy.signal import butter, lfilter, filtfilt, hilbert, firwin
    from utils.filters import magic_filter_taps, custom_fir

    b, a = butter(4, [8 / nq, 12 / nq], btype='band')

    x = filtfilt(b, a, data[:, 15])

    amplitude_envelope = np.abs(hilbert(x))

    env = RCEnvelopeDetector(0.95)
    y = np.array([env.get_envelope(x_) for x_ in x])

    plt.plot(x, alpha=0.2)
    plt.plot(amplitude_envelope)
    plt.plot(y)
    plt.show()


def find_lag(x, target, fs, show=False):
    n = 1000
    nor = lambda x:  (x - np.mean(x)) / np.std(x)
    lags = np.arange(n)
    mses = np.zeros_like(lags).astype(float)
    n_points = len(target) - n
    for lag in lags:
        mses[lag] = np.mean((nor(target[:n_points]) - nor(x[lag:n_points+lag]))**2)
    lag = np.argmin(mses)

    if show:
        f, (ax1, ax2) = plt.subplots(2)
        ax1.plot(mses)

        ax1.plot(lag, np.min(mses), 'or')
        ax1.text(lag+n//100*2, np.min(mses), '{} ({:.3f} s)'.format(lag, lag/fs))
        ax2.plot(nor(target))

        ax2.plot(nor(x[lag:]), alpha=1)
        ax2.plot(nor(x), alpha=0.5)
        ax2.legend(['target',  'x[{}:]'.format(lag), 'x'])
        plt.show()
    return lag