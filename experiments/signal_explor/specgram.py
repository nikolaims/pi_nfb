from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter

from utils.data.loaders import get_signal
from utils.filters.ideal_filter import get_fir_filter
from utils.filters.main_freq import get_main_freq

fs = 250
x = get_signal()[:, 4]
main_freq = get_main_freq(x, fs, band=(8, 12))
i_taps, i_delay = get_fir_filter(fs, main_freq, order=2000, show=0)
x = lfilter(i_taps, [1.], x)[i_delay:]
f, t, Sxx = signal.spectrogram(x, fs)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()