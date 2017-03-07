import h5py
import numpy as np
import pylab as plt

from scipy.signal import hilbert, lfilter
from utils.data.loaders import get_signal_data
from utils.envelope import find_lag
from utils.filters.ideal_filter import get_fir_filter
from utils.filters.main_freq import get_main_freq
from utils.filters import dc_blocker

# parameters
fs = 250
ch = 4
i_fir_order = 2000 # ideal FIR order
fir_order = 400
fir2_order = fir_order*2
fir2_width = 6

# load signal
raw, signals, derived = get_signal_data()
raw = signals[5000:25000]




# test data
data_dir = 'C:\\Users\\Nikolai\\Downloads\\pilot_5Days_Rakhmankulov_Day1_02-27_17-27-34\\'
with h5py.File(data_dir + 'experiment_data.h5', 'r') as f:  # TODO: path
    protocol = 'protocol6'
    raw = f[protocol+'/raw_data'][:][:, 0]
    derived = f[protocol+'/signals_data'][:][:, 0]
#raw = dc_blocker(raw)

plt.plot(raw)
plt.show()



#plt.plot(signals)
#plt.plot(derived)

plt.show()


# get main freq
main_freq = get_main_freq(raw, fs, band=(8, 12))

band = (main_freq - 1, main_freq + 1)
band=(9,14)
#main_freq=20
# ideal FIR filter
i_taps, i_delay = get_fir_filter(fs, main_freq, order=i_fir_order, show=0, band=band)
i_signal = lfilter(i_taps, [1.], raw)[i_delay:]
i_envelope = np.abs(hilbert(i_signal))
plt.plot(i_signal, alpha=0.2)





from utils.filters.fft_chunk import fft_chunk_envelope

filtered = fft_chunk_envelope(raw, band, fs)
nor = lambda x: (x - np.mean(x)) / np.std(x)
plt.plot(nor(i_envelope), 'b')
plt.plot(nor(filtered[:]))
plt.plot(nor(derived))
plt.show()

lag = find_lag(derived, i_envelope, fs/2, True)
print(lag)
derived_envelope = nor(derived[lag:])
i_envelope = nor(i_envelope)


slc = slice(15000, 25000)
i_var = np.var(i_envelope[slc])*0 + 1
i_cent = i_envelope[slc] - np.mean(i_envelope[slc])

metric = lambda x: np.mean((nor(i_envelope[slc]) - nor(x[slc])) ** 2) / i_var

corr_metric = lambda x: np.corrcoef(i_cent, x[slc] - np.mean(x[slc]))[0, 1]

print('metrics', metric(derived_envelope), corr_metric(derived_envelope))