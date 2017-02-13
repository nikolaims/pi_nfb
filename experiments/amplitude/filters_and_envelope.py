import pylab as plt
from scipy.signal import *
from utils.data.loaders import get_ideal_signal, load_feedback, get_signal
from utils.filters import magic_filter_taps, min_phase_magic_filter
from utils.sinbase import get_base
from itertools import combinations
import numpy as np

n = 10000
n_components = 50
fs = 250
band = (8, 12)
time   = np.arange(n) / fs

# load signal
signal = get_signal()[:n, 15]


# IIR filt filt
w = 0.1
gain = [0, 0, 1, 1, 0, 0]
taps = firwin2(1000, [0 , band[0]-w, band[0], band[1], band[1]+w, fs/2], gain, nyq=fs/2)
ideal = filtfilt(taps, 1, signal)
plt.plot(np.abs(ideal), 'b', alpha=0.6)
plt.plot(np.abs(hilbert(ideal)), 'b')


# fft
from scipy.fftpack import rfft, irfft, fftfreq
W = fftfreq(signal.size, d=1/fs*2)
f_signal = rfft(signal)
cut_f_signal = f_signal.copy()
cut_f_signal[(W<8) | (W>12)] = 0
cut_signal = irfft(cut_f_signal)
plt.plot(np.abs(cut_signal), 'k', alpha=0.6)
plt.plot(np.abs(hilbert(cut_signal)), 'k')
print(np.mean((np.abs(hilbert(cut_signal)) - np.abs(hilbert(ideal)))**2)/np.var(np.abs(hilbert(cut_signal))))


# fir minphase
fir_signal = lfilter(min_phase_magic_filter(), 1, signal)[28:]
plt.plot(np.abs(fir_signal), 'g', alpha=0.6)
plt.plot(np.abs(hilbert(fir_signal)), 'g')

# iir fir
fir_signal = lfilter(magic_filter_taps(), 1, signal)[28:]
plt.plot(np.abs(fir_signal), 'r', alpha=0.6)
plt.plot(np.abs(hilbert(fir_signal)), 'r')



plt.show()