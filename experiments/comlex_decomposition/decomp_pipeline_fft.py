from utils.data.loaders import load_normalised_raw_signal
import numpy as np
import pylab as plt
from scipy.signal import welch, lfilter, savgol_coeffs, butter, minimum_phase
import seaborn as sns
cm = sns.color_palette()
sns.set_style('white')

from utils.filters.fft_chunk import fft_chunk_envelope
from utils.filters.main_freq import get_main_freq
from utils.metrics import find_lag
from utils.pipeline.ideal_envelope_detector import ideal_envelope_detector
nor = lambda x: (x - x.mean()) / x.std()


fs = 250
raw = load_normalised_raw_signal()
n = 25000
# get main freq
fn = 1.5
main_freq = get_main_freq(raw, fs, band=(8, 12))
band = (main_freq - fn, main_freq + fn)

# ideal signal
i_signal, i_envelope = ideal_envelope_detector(raw, band, fs, n)

# truncate samples
raw = raw[:n]

#sine
sine = np.exp(-1j*np.arange(25000)/fs*2*np.pi*(main_freq))



# plot spec
f = plt.figure(figsize=(5, 3))
ff, v = welch(raw, fs, nperseg=1000,  return_onesided=False, detrend=False)
ff = np.hstack((ff[len(ff)//2:], ff[:len(ff)//2]))
v = np.hstack((v[len(v)//2:], v[:len(v)//2]))
plt.plot(ff, v, c=cm[0])
plt.fill_between(*welch(i_signal, fs, nperseg=1000,  return_onesided=False, detrend=False), color=cm[0], alpha=0.8)

ff, v = welch(raw*sine, fs, nperseg=1000,  return_onesided=False, detrend=False)
ff = np.hstack((ff[len(ff)//2:], ff[:len(ff)//2]))
v = np.hstack((v[len(v)//2:], v[:len(v)//2]))
plt.plot(ff, v, c=cm[1])

ff, v = welch(i_signal*sine, fs, nperseg=1000,  return_onesided=False, detrend=False)
ff = np.hstack((ff[len(ff)//2:], ff[:len(ff)//2]))
v = np.hstack((v[len(v)//2:], v[:len(v)//2]))
plt.fill_between(ff, v, color=cm[1], alpha=0.8)
plt.xlabel('f, $Hz$')
plt.ylabel('PSD')
plt.xlim(-15, 15)
plt.ylim(0, 0.05)
f.tight_layout()
f.savefig('spec_shift.png', dpi=200)
plt.show()

# plot how signal transformed
f = plt.figure(figsize=(5, 5))
key = 'raw'
i_signal = 0.8*{'raw': raw, 'ideal': i_signal}[key]

smoothed, filtered, window = fft_chunk_envelope(i_signal, band=(main_freq - fn, main_freq + fn), fs=fs, smoothing_factor=0.1, chunk_size=1)

plt.plot(i_signal*0, c=cm[0], alpha=0.5)
start_index = 8500
rrange = np.arange(start_index, start_index+len(window)//2)
plt.plot(i_signal[:start_index], c=cm[0], alpha=1)
plt.plot(np.arange(start_index + len(window)//2, len(i_signal)), i_signal[start_index + len(window)//2:], c=cm[0], alpha=1)
plt.fill_between(rrange, 2*window[:len(window)//2], color=cm[1], alpha=0.8)
plt.plot(rrange, i_signal[rrange], color=cm[0], alpha=0.5)


plt.plot(2*filtered - 4, c=cm[2], alpha=1)
plt.plot(2*filtered*0 - 4, c=cm[2], alpha=0.5)

b, a = butter(1, 1.5/fs*2, )
iir_filtered = 2*lfilter(b, a, i_signal * sine)
plt.plot(2*smoothed - 8, c=cm[3], alpha=1)
plt.plot(2*smoothed*0 - 8, c=cm[3], alpha=0.5)

plt.xlim(8500, 9100)
plt.ylim(-11, 2)
plt.axis('off')
f.savefig('pipeline_{}.png'.format(key), dpi=200)
plt.show()