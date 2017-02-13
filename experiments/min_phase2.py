import numpy as np
from scipy import signal, fftpack
import matplotlib.pyplot as plt

from mne.time_frequency.tfr import morlet
from mne1.viz import plot_filter, plot_ideal_filter

import mne

sfreq = 250
f_p = 40.
flim = (1., sfreq / 2.)  # limits for plotting

nyq = sfreq / 2.  # the Nyquist frequency is half our sample rate
freq = [0, f_p, f_p, nyq]
gain = [0, 1, 1, 0, 0]

third_height = np.array(plt.rcParams['figure.figsize']) * [1, 1. / 3.]
ax = plt.subplots(1, figsize=third_height)[1]
n = 50
t = np.arange(-n // 2, n // 2) / sfreq  # center our sinc
#h = signal.firwin2(n, freq, gain, nyq=nyq)
trans_bandwidth = 25
f_s = f_p + trans_bandwidth
gain = [0, 1, 1, 0, 0]
freq = [0 , 8, 12, 18, nyq]

h = signal.firwin2(n, freq, gain, nyq=nyq)
plot_filter(h, sfreq, freq, gain, 'Windowed 50-Hz transition (0.2 sec)',
            flim=flim, fscale='linear')

from mne1.fixes import minimum_phase
h_min = minimum_phase(h)
plot_filter(h_min, sfreq, freq, gain, 'Minimum-phase', flim=flim)