from scipy.signal import lfilter, hilbert
import numpy as np
from utils.data.loaders import get_signal
import pylab as plt

from utils.filters.ideal_filter import get_fir_filter
from utils.filters.main_freq import get_main_freq
from mne1.filter import create_filter
from mne1.viz import plot_filter

fs = 250
raw = get_signal()[:, 4]

# get main freq
main_freq = get_main_freq(raw, fs, band=(8, 12))

# ideal FIR filter
i_taps, i_delay = get_fir_filter(fs, main_freq, order=2000, show=0)
i_signal = lfilter(i_taps, [1.], raw)[i_delay:]
i_envelope = np.abs(hilbert(i_signal))


#x = i_signal
min_phase_taps = create_filter(raw, 250, main_freq - 1, main_freq + 1,  filter_length=2000, phase='minimum', l_trans_bandwidth=1)
plot_filter(min_phase_taps, 250)
x = lfilter(min_phase_taps, [1], raw)

min_phase_taps = create_filter(np.abs(x), 250, None, 10,  filter_length=100, phase='minimum')
plot_filter(min_phase_taps, 250)
m_signal = lfilter(min_phase_taps, [1], np.abs(x))

n = lambda x: (x - x.mean())/x.std()
plt.plot(n(i_envelope), 'b')

plt.plot(n(m_signal), 'g', alpha=1)
plt.show()
