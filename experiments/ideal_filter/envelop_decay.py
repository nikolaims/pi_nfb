from collections import OrderedDict
from pyexpat import errors

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
taps = firwin2(order, freq, gain, nyq=fs/2)
plot_filter(taps, fs, freq, gain, flim=(5, 15), fscale='linear')


# load raw signal
signal = get_signal()[:, 15]

# filter signal
signals = OrderedDict()
signals['raw'] = signal
y = lfilter(taps, [1], signal)[order // 2:order // 2 + 20000]


n_orders = 100
orders = np.linspace(20, 500, n_orders, dtype=int)
print(orders)
errors = np.zeros(n_orders)

y_h = np.abs(hilbert(y))
for k, order in enumerate(orders):
    taps = firwin2(order, freq, gain, nyq=fs / 2)
    x = lfilter(taps, [1], signal)[order // 2:order // 2 + 20000]
    errors[k] = np.mean((y_h - np.abs(hilbert(x)))**2)
    #plt.plot(np.abs(hilbert(y)))
    #plt.plot(np.abs(hilbert(x)))
    #plt.show()
errors /= np.var(y_h)

plt.plot(orders, errors, '.-')
plt.ylabel('$mse( FIR_{2000} - FIR_{K} )/ var( FIR_{2000} )$')
plt.xlabel('$K$')
plt.show()


plt.plot(y_h)
orders = [50, 100, 200, 400][::-1]
legend = ['$FIR_{2000}$']
for order in orders:
    taps = firwin2(order, freq, gain, nyq=fs / 2)
    x = lfilter(taps, [1], signal)[order // 2:order // 2 + 20000]
    x = np.abs(hilbert(x))
    plt.plot(x)
    legend += ['$FIR_{' + str(order) + '}$']

plt.legend(legend)
plt.show()