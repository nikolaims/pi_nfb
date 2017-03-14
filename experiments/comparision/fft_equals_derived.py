from scipy.signal import lfilter, hilbert

from utils.data.loaders import get_signal_data
import numpy as np
import pylab as plt
from utils.filters.fft_chunk import fft_chunk_envelope
from utils.filters.ideal_filter import get_fir_filter
from utils.filters.main_freq import get_main_freq

fs = 250
data, raw, derived = get_signal_data()

# main_freq =  get_main_freq(raw, fs, band=(9, 14), show=0)
nor = lambda x: (x - x.mean())/x.std()
i_taps, i_delay = get_fir_filter(fs, None, show=0, band=(9, 14))
i_signal = lfilter(i_taps, [1.], raw)[i_delay:]
i_envelope = np.abs(hilbert(i_signal))
fft_envelope = fft_chunk_envelope(raw, band=(9, 14), fs=fs)
plt.plot(i_signal)
plt.plot(nor(i_envelope))
plt.plot(nor(derived))
plt.plot(nor(fft_envelope))
plt.show()