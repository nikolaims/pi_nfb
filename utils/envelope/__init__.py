class RCEnvelopeDetector:
    def __init__(self, factor):
        self.factor = factor
        self.y = 0

    def get_envelope(self, x):
        x = abs(x)
        self.y = x if x > self.y else self.y * self.factor + x * (1 - self.factor)
        return self.y


if __name__ == '__main__':
    import pylab as plt
    from utils.data.loaders import load_feedback
    from utils.filters import dc_blocker
    import numpy as np

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
