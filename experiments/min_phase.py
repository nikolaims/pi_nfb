import pylab as plt
from scipy.signal import *
from utils.data.loaders import get_ideal_signal, load_feedback, get_signal
from utils.filters import magic_filter_taps, custom_fir
from utils.sinbase import get_base
from itertools import combinations
import numpy as np

n = 10000
n_components = 50
fs = 250
band = (8, 12)
time   = np.arange(n) / fs

# load signal
signal = get_signal()[:n, 15]

# fir
taps = custom_fir(fs=250, width=6, ripple_db=40, band=(8, 12))[0]#  magic_filter_taps()
fir_signal = lfilter(taps, 1, signal)[47:]
plt.plot(fir_signal)


def minimum_phase(h):
    """Convert a linear-phase FIR filter to minimum phase.
    Parameters
    ----------
    h : array
        Linear-phase FIR filter coefficients.
    Returns
    -------
    h_minimum : array
        The minimum-phase version of the filter, with length
        ``(length(h) + 1) // 2``.
    """
    try:
        from scipy.signal import minimum_phase
    except Exception:
        pass
    else:
        return minimum_phase(h)
    from scipy.fftpack import fft, ifft
    h = np.asarray(h)
    if np.iscomplexobj(h):
        raise ValueError('Complex filters not supported')
    if h.ndim != 1 or h.size <= 2:
        raise ValueError('h must be 1D and at least 2 samples long')
    n_half = len(h) // 2
    if not np.allclose(h[-n_half:][::-1], h[:n_half]):
        print('h does not appear to by symmetric, conversion may '
                      'fail')
    n_fft = 2 ** int(np.ceil(np.log2(2 * (len(h) - 1) / 0.01)))
    # zero-pad; calculate the DFT
    h_temp = np.abs(fft(h, n_fft))
    # take 0.25*log(|H|**2) = 0.5*log(|H|)
    h_temp += 1e-7 * h_temp[h_temp > 0].min()  # don't let log blow up
    np.log(h_temp, out=h_temp)
    h_temp *= 0.5
    # IDFT
    h_temp = ifft(h_temp).real
    # multiply pointwise by the homomorphic filter
    # lmin[n] = 2u[n] - d[n]
    win = np.zeros(n_fft)
    win[0] = 1
    stop = (len(h) + 1) // 2
    win[1:stop] = 2
    if len(h) % 2:
        win[stop] = 1
    h_temp *= win
    h_temp = ifft(np.exp(fft(h_temp)))
    h_minimum = h_temp.real
    n_out = n_half + len(h) % 2
    return h_minimum[:n_out]


min_taps = minimum_phase(taps)
fir_signal = lfilter(min_taps, 1, signal)[26:]
plt.plot(fir_signal)

b, a = butter(6, [band[0]/fs*2, band[1]/fs*2], btype='band')
ideal = filtfilt(b, a, signal)
plt.plot(ideal)

plt.legend(['FIR - delay(47)', 'min phase FIR - delay(26)', 'filtfilt butter(6)'])


plt.show()


nq = 125
w, h = freqz(taps, worN=8000)
plt.plot((w/np.pi)*nq, np.absolute(h), linewidth=2)
w, h = freqz(min_taps, worN=8000)
plt.plot((w/np.pi)*nq, np.absolute(h), linewidth=2)

plt.legend(['FIR - delay(47)', 'min phase FIR - delay(26)'])
plt.show()



w, h = group_delay((taps, [1]))
plt.plot((w/np.pi)*nq, np.absolute(h), linewidth=2)
w, h = group_delay((min_taps, [1]))
plt.plot((w/np.pi)*nq, np.absolute(h), linewidth=2)
plt.legend(['FIR - delay(47)', 'min phase FIR - delay(26)'])
plt.show()