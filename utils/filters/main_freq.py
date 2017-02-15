import numpy as np
import pylab as plt
from scipy.signal import periodogram

def get_main_freq(x, fs, band, show=False):
    f, Pxx = periodogram(x, fs=fs, nfft=fs*10)
    if show:
        plt.plot(f, Pxx)
    Pxx[(f<band[0]) | (f>band[1])] = 0
    f_alpha = f[np.argmax(Pxx)]
    print('Argmax(Pxx[band]): {:.2f} Hz'.format(f_alpha))
    if show:
        plt.plot([f_alpha],[np.max(Pxx)],'ro')
        plt.show()
    return f_alpha



if __name__ == '__main__':
    from utils.data.loaders import get_signal
    signal = get_signal()[:, 15]
    get_main_freq(x=signal, fs=250, band=(8,12), show=True)
