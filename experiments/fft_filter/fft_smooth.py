import numpy as np
import pylab as plt
from scipy.signal import hilbert, lfilter, welch
from utils.data.loaders import load_normalised_raw_signal
from utils.envelope.smoothers import exp_smooth
from utils.filters.fft_chunk import fft_chunk_envelope
from utils.filters.ideal_filter import get_fir_filter
from utils.filters.main_freq import get_main_freq
from utils.metrics import find_lag
from utils.rls.naive import naive_rls
from utils.sinbase import get_base_by_freqs

fs = 250
n = 20000
t = np.arange(n)/fs

# load signal
raw = load_normalised_raw_signal()

# get main freq
main_freq = 10#get_main_freq(raw, fs, band=(8, 12))
band = (main_freq - 1, main_freq + 1)

# ideal signal
i_taps, i_delay = get_fir_filter(fs, None, show=0, band=band)
i_signal = lfilter(i_taps, [1.], raw)[i_delay:][:n]
i_envelope = np.abs(hilbert(i_signal))

# truncate samples
raw = raw[:n]

# fft envelope
fft_envelope = fft_chunk_envelope(raw, band=band, fs=fs, smoothing_factor=1, chunk_size=1, n_samples=500)
freqs = [20]
base = get_base_by_freqs(n, fs, freqs)/(len(freqs)*2)
p, w, w_path = naive_rls(base, fft_envelope, mu=1)
fft_envelope_without_ring = fft_envelope - p
fft_envelope_without_ring_exp = exp_smooth(fft_envelope - p, 0.3)
find_lag(fft_envelope_without_ring_exp, i_envelope, show=True, fs=fs)
plt.plot(*welch(fft_envelope_without_ring, fs))
plt.plot(*welch(fft_envelope, fs))
#plt.plot(*welch(fft_envelope_without_ring_exp, fs))
plt.legend(['fft', 'fft - 20Hz rls'])
plt.show()

# classic fft
fft_envelope1 = fft_chunk_envelope(raw, band=band, fs=fs, n_samples=500, chunk_size=1, smoothing_factor=1)
fft_envelope1 = exp_smooth(fft_envelope1, 0.3)
find_lag(fft_envelope, i_envelope, show=True, fs=fs)


nor = lambda x: (x - x.mean()) / x.std()
#plt.plot(raw, 'k', alpha=0.5)
#plt.plot(i_signal, 'b', alpha=0.5)
print(nor(i_envelope).shape)
import seaborn
seaborn.set_style('white')
plt.plot(t, nor(i_envelope))
plt.plot(t, nor(fft_envelope), alpha=0.5)
plt.plot(t, nor(fft_envelope1))
plt.plot(t, nor(fft_envelope_without_ring_exp), 'k')
plt.legend(['ideal', 'fft', 'fft + exp smooth', 'fft - 20Hz rls + exp smooth'])
plt.show()