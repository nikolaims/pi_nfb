from collections import OrderedDict

import numpy as np
import pylab as plt
from scipy.signal import *
from utils.data.loaders import get_signal
from utils.filters.fft import fft_filter
from mne1.viz import plot_filter
#from seaborn import color_palette
cm = ['k', 'b', 'g', 'r', 'm']


# design ideal filter
order = 2000
band = (9, 11)
w = 0.1
fs = 250
freq = [0, band[0]-w, band[0], band[1], band[1]+w, fs/2]
gain = [0, 0, 1, 1, 0, 0]
taps = firwin2(order, freq, [0, 0, 1, 1, 0, 0], nyq=fs/2)
plot_filter(taps, fs, freq, gain, flim=(5, 15), fscale='linear')


# load raw signal
signal = get_signal()[:, 15]

# filter signal
signals = OrderedDict()
signals['raw'] = signal
signals['lfilter - delay'] = lfilter(taps, [1], signal)[order // 2:]
signals['filtfilt'] = filtfilt(taps, [1], signal)
signals['fft'] = fft_filter(signal, band, fs)

legend = []
for c, (key, x) in enumerate(signals.items()):
    plt.plot(x, c=cm[c], alpha=0.5 if key == 'raw' else 0.8)
    legend += [key]
    if not key == 'raw':
        plt.plot(np.abs(hilbert(x)), c=cm[c])
        legend += [key + ' envelope']
plt.legend(legend)

plt.show()
