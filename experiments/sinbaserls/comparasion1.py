import pylab as plt
from scipy.signal import group_delay
from scipy.signal import hilbert, lfilter, filtfilt

from mne1.filter import create_filter
from utils.data.loaders import get_signal
from utils.envelope import find_lag
from utils.filters import dc_blocker
from utils.filters.fft_chunk import fft_chunk_envelope
from utils.filters.ideal_filter import get_fir_filter, get_fir_filter_high_pass
from utils.filters.main_freq import get_main_freq
from utils.filters.min_phase import get_filter_delay_at_freq, minimum_phase
from utils.filters.sin_base_rls import sin_base_rls_filter
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
fir2_taps, fir2_delay = get_fir_filter(fs, main_freq, order=fir2_order, show=0, width=fir2_width)#, band=[main_freq-3, main_freq+100])
#fir2_taps, fir2_delay = get_fir_filter_high_pass(fs, main_freq=main_freq, order=101, width=1, show=1)





#222222222222

min_phase_taps = create_filter(raw, 250, main_freq, None,  filter_length=500, phase='minimum')


#min_phase_delay = get_filter_delay_at_freq(min_phase_taps, main_freq, fs) + 10
#print('Min phase delay: ', min_phase_delay)
min_phase_signal = lfilter(min_phase_taps, [1.], raw)#[min_phase_delay:]

lag = find_lag(np.abs(hilbert(filtfilt(i_taps, [1.], min_phase_signal))), i_envelope, 250, show=0)
print('Min phase delay: ', lag)
min_phase_signal = min_phase_signal#[lag:]
#min_phase_signal = dc_blocker(raw,r=0.9)
min_phase_envelope = np.abs(hilbert(min_phase_signal))
from mne1.viz import plot_filter
plot_filter(min_phase_taps, 250, fscale='linear', flim=(0, 20))

# rls fitting




rls_signal2, rls_envelope2 = sin_base_rls_filter(min_phase_signal, fs, main_freq, n_components=32)



rls_envelope = rls_envelope2#rls_envelope[:25000]*0.4 + rls_envelope2[:25000]*0.6
rls_signal = rls_signal2
print('RLS lag', find_lag(rls_envelope[:25000], i_envelope[:25000], 250, 1))

if 1:
    import seaborn as sns

    sns.set_style("white")
    sns.despine()
    fft_envelope = fft_chunk_envelope(raw, band=(main_freq-1, main_freq+1), fs=fs, smoothing_factor=0.1, chunk_size=1)
    cm = sns.color_palette()
    plt.plot(i_signal**2, c=cm[0], alpha=0.5)
    plt.plot(i_envelope**2, c=cm[0], alpha=1, linewidth=2)
    plt.plot(rls_signal**2, c=cm[1], alpha=0.5)
    plt.plot(rls_envelope**2, c=cm[1], alpha=1, linewidth=2)
    plt.plot(fft_envelope**2, c=cm[2], alpha=1, linewidth=2)
    plt.legend(['$X^2(n)$','$P(n)$','$X_{RLS}^2(n)$','$P_{RLS}(n)$','$P_{FFT}(n)$'])
    plt.xlabel('$n$, [samples]')
    plt.ylabel('$magnitude^2$')
    plt.show()

# plot signals and envelopes
if 1:
    print('wow')
    plt.plot(i_signal, 'b', alpha=1)
    plt.plot(i_envelope, 'b', alpha=0.8, linewidth=2)
    #plt.plot(fir_signal, 'g')
    #plt.plot(fir_envelope, 'g')
    plt.plot(min_phase_signal, 'r', alpha=0.6)
    #plt.plot(min_phase_envelope, 'r', alpha=0.5, linewidth=2)
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