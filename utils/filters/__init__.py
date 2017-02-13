from numpy import zeros_like
from scipy.signal import kaiserord, firwin
from scipy.signal import *
import numpy as np

def dc_blocker(x, r=0.99):
    # DC Blocker https://ccrma.stanford.edu/~jos/fp/DC_Blocker.html
    y = zeros_like(x)
    for n in range(1, x.shape[0]):
        y[n] = x[n] - x[n-1] + r * y[n-1]
    return y

def magic_filter_taps(bad=False):
    # nyquist rate
    nq = 250 / 2

    # transition width
    width = 7 / nq

    # attenuation in the stop band [db]
    ripple_db = 30.0

    if bad:
        # nyquist rate
        nq = 250 / 2

        # transition width
        width = 11 / nq

        # attenuation in the stop band [db]
        ripple_db = 25.0

    # compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)
    print('FIR filter order:', N)

    # create a bandpass FIR filter
    taps = firwin(N, [7 / nq, 13 / nq], window=('kaiser', beta), pass_zero=False)
    return taps


def custom_fir(fs=250, width=7, ripple_db=30, band=(7, 13)):
    # nyquist rate
    nq = fs/2

    # transition width
    width = width / nq

    # attenuation in the stop band [db]
    ripple_db = ripple_db

    # compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)
    print('FIR filter delay:', N//2)

    # create a bandpass FIR filter
    taps = firwin(N, [band[0] / nq, band[1] / nq], window=('kaiser', beta), pass_zero=False)
    return taps, N//2


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


def min_phase_magic_filter():
    fs = 250
    band = (8, 12)
    w = 2
    gain = [0, 0, 1, 1, 0, 0]
    freq = [0, band[0] - w, band[0], band[1], band[1] + w, fs / 2]
    taps = firwin2(400, freq, gain, nyq=fs / 2)
    min_taps = minimum_phase(taps)
    return min_taps
