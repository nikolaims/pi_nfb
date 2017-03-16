from utils.data.loaders import load_normalised_raw_signal
import numpy as np
import pylab as plt
from scipy.signal import welch

from utils.filters.main_freq import get_main_freq
from utils.pipeline.ideal_envelope_detector import ideal_envelope_detector
nor = lambda x: (x - x.mean()) / x.std()

fs = 250
raw = load_normalised_raw_signal()
n = 25000
# get main freq
main_freq = get_main_freq(raw, fs, band=(8, 12))
band = (main_freq - 1, main_freq + 1)

# ideal signal
i_signal, i_envelope = ideal_envelope_detector(raw, band, fs, n)

# truncate samples
raw = raw[:n]


plt.plot(raw)
plt.plot(i_signal)
plt.plot(i_envelope)
plt.show()

sine = np.sin(np.arange(25000)/fs*2*np.pi*(main_freq))
#sine = np.exp(-1j*np.arange(25000)/fs*2*np.pi*(main_freq))


plt.plot(*welch(raw * sine, fs))
plt.plot(*welch(raw, fs))
plt.show()

from utils.filters.fft import fft_filter
from scipy.signal import firwin2, lfilter
am = fft_filter(np.real(raw * sine), (0, 2), fs)
ft = 1
w=0.1
order=100
freq = [0, ft-w, ft+w, fs/2]
gain = [1, 1, 0, 0]
taps = firwin2(order, freq, gain, nyq=fs / 2)
am1 = lfilter(taps, [1.], raw * sine)

plt.plot(nor(i_envelope))
plt.plot(nor(am))
plt.show()

