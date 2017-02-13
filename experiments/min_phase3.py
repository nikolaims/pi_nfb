import pylab as plt
from mne1.viz import plot_filter
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
w = 2
gain = [0, 0, 1, 1, 0, 0]
freq = [0 , band[0]-w, band[0], band[1], band[1]+w, fs/2]

f_n = 400
taps = firwin2(f_n, freq, gain, nyq=fs/2)
fir_signal = lfilter(taps, 1, signal)[f_n//2:]
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
fir_signal = lfilter(min_taps, 1, signal)[28:]
plt.plot(fir_signal)

tapsi = firwin2(2000, freq, gain, nyq=fs/2)
ideal = filtfilt(tapsi, 1, signal)
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



w, h = group_delay((taps, 1))
plt.plot((w/np.pi)*nq, h, linewidth=2)
w, h = group_delay((min_taps, 1))
plt.plot((w/np.pi)*nq, h, linewidth=2)
plt.ylim(0, f_n + 10)
plt.legend(['FIR - delay(47)', 'min phase FIR - delay(26)'])
plt.show()


flim = (1., 20)
fa = plot_filter(min_taps, 250, freq, gain, 'Windowed 50-Hz transition (0.2 sec)',
            flim=flim, fscale='linear', color='g', show=False)
plot_filter(firwin2(f_n//2, freq, gain, nyq=fs/2), 250, freq, gain, 'Windowed 50-Hz transition (0.2 sec)',
            flim=flim, fscale='linear', color='b', show=False, fig_axes=fa )

plot_filter(taps, 250, freq, gain, 'Filters',
            flim=flim, fscale='linear', color='r', show=False, fig_axes=fa )

plt.legend(['FIR 400 taps', 'FIR 200 taps', 'Min-phase FIR 200 taps'])

plt.show()