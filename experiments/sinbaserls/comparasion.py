import pylab as plt
from scipy.signal import group_delay
from scipy.signal import hilbert, lfilter, filtfilt

from utils.data.loaders import get_signal
from utils.filters.ideal_filter import get_fir_filter
from utils.filters.main_freq import get_main_freq
from utils.filters.min_phase import get_filter_delay_at_freq, minimum_phase
from utils.sinbase import get_base
from itertools import combinations
import numpy as np

# parameters
fs = 250
ch = 15
i_fir_order = 2000 # ideal FIR order
fir_order = 400
fir2_order = fir_order*2
fir2_width = 6

# load signal
raw = get_signal()[:, ch]

# get main freq
main_freq = get_main_freq(raw, fs, band=(8, 12))

# ideal FIR filter
i_taps, i_delay = get_fir_filter(fs, main_freq, order=i_fir_order, show=0)
i_signal = lfilter(i_taps, [1.], raw)[i_delay:]
i_envelope = np.abs(hilbert(i_signal))

# FIR filter
fir_taps, fir_delay = get_fir_filter(fs, main_freq, order=fir_order, show=0, width=1)
fir_signal = lfilter(fir_taps, [1.], raw)[fir_delay:]
fir_envelope = np.abs(hilbert(fir_signal))

# min_phase filter
fir2_taps, fir2_delay = get_fir_filter(fs, main_freq, order=fir2_order, show=0, width=fir2_width)
min_phase_taps = minimum_phase(fir2_taps)
min_phase_delay = get_filter_delay_at_freq(min_phase_taps, main_freq, fs)+13
print('Min phase delay: ', min_phase_delay)
min_phase_signal = lfilter(min_phase_taps, [1.], raw)[min_phase_delay:]
min_phase_envelope = np.abs(hilbert(min_phase_signal))
#from mne1.viz import plot_filter
#plot_filter(min_phase_taps, 250, fscale='linear', flim=(0, 20))

# rls fitting
def rls_filter(x, n_components=16):
    n = len(x)
    ww = 0.9
    base, freqs = get_base(n, fs=fs, n_components=n_components, low=main_freq - ww, high=main_freq + ww, nonlinear=1.25)
    # prepare sin
    projections = []
    pairs = lambda: combinations(range(2 * n_components), 2)
    for i, j in pairs():
        freq = (1 - 2 * (i % 2)) * np.pi / 2 * ((i - j) % 2) + (freqs[j // 2] - freqs[i // 2]) * 2 * np.pi * np.arange(
            n) / fs
        projections.append(2 * np.cos(freq))
    projections = np.array(projections + [np.ones((n,)) for k in range(2 * n_components)])
    c_ind = np.array(list(pairs()) + [(k, k) for k in range(2 * n_components)]).T
    # rls
    from utils.models.rls import DelayedRLSPredictor
    rls = DelayedRLSPredictor(2 * n_components, mu=1, M=1, delay=0, lambda_=0.9991)
    rls_signal = np.zeros_like(x)
    rls_envelope = np.zeros_like(x)
    ws = np.zeros((n, 2 * n_components))
    for k in range(n):
        rls_signal[k] = rls.predict(base[k], x[k])
        w = rls._w
        ws[k] = w
        rls_envelope[k] = (np.sum(projections[:, k] * w[c_ind[0]] * w[c_ind[1]])) ** 0.5
    return rls_signal, rls_envelope


rls_signal, rls_envelope = rls_filter(min_phase_signal)


# plot signals and envelopes
if 1:
    print('wow')
    plt.plot(i_signal, 'b', alpha=1)
    plt.plot(i_envelope, 'b', alpha=0.8, linewidth=2)
    #plt.plot(fir_signal, 'g')
    #plt.plot(fir_envelope, 'g')
    plt.plot(min_phase_signal, 'r', alpha=1)
    plt.plot(min_phase_envelope, 'r', alpha=0.8, linewidth=2)
    plt.plot(rls_signal, 'k', alpha=1)
    plt.plot(rls_envelope, 'k', alpha=0.8, linewidth=2)


    from sklearn.metrics import mean_squared_error
    slc = slice(15000, 25000)
    i_var = np.var(i_envelope[slc])
    i_cent = i_envelope[slc] - np.mean(i_envelope[slc])

    metric = lambda x: np.mean((i_envelope[slc] - x[slc])**2)/i_var

    corr_metric = lambda x: np.corrcoef(i_cent, x[slc] - np.mean(x[slc]))[0, 1]
    print('min_phase', metric(min_phase_envelope), corr_metric(min_phase_envelope))
    print('rls', metric(rls_envelope), corr_metric(rls_envelope))
    plt.show()