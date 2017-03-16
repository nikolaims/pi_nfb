from utils.data.loaders import load_normalised_raw_signal
import numpy as np
import pylab as plt
from scipy.signal import welch

from utils.filters.main_freq import get_main_freq
from utils.metrics import find_lag
from utils.pipeline.ideal_envelope_detector import ideal_envelope_detector
nor = lambda x: (x - x.mean()) / x.std()

fs = 250
raw = load_normalised_raw_signal()
n = 25000
# get main freq
fn = 2
main_freq = get_main_freq(raw, fs, band=(8, 12))
band = (main_freq - fn, main_freq + fn)

# ideal signal
i_signal, i_envelope = ideal_envelope_detector(raw, band, fs, n)

# truncate samples
raw = raw[:n]


plt.plot(raw)
plt.plot(i_signal)
plt.plot(i_envelope)
plt.show()

sine = np.sin(np.arange(25000)/fs*2*np.pi*(main_freq))
sine = np.exp(-1j*np.arange(25000)/fs*2*np.pi*(main_freq))


plt.plot(*welch(raw * sine, fs))
plt.plot(*welch(raw, fs))
plt.show()

from utils.filters.fft import fft_filter
from scipy.signal import firwin2, lfilter, butter
am = fft_filter(np.real(raw * sine), (0, 2), fs)
w=0
order=150
freq = [0, fn, fn+w, fs/2]
gain = [1, 1, 0, 0]
taps = firwin2(order, freq, gain, nyq=fs / 2)
am1 = lfilter(taps, [1.], i_signal * sine)[order//2:]
from mne1.filter import minimum_phase, create_filter
from mne1.viz import plot_filter
mne_taps = minimum_phase(taps)
#mne_taps = create_filter(raw, fs, None, 0.5, phase='minimum', h_trans_bandwidth=1.5, filter_length=100)
plot_filter(mne_taps, fs)
#b, a = butter(3, 2/fs*2, )
b, a = mne_taps, [1.]
am2 = np.abs(2*lfilter(b, a, raw * sine))
find_lag(am2, i_envelope, fs, show=True)

plt.plot(i_envelope)
plt.plot(np.abs(2*am1))
plt.show()

