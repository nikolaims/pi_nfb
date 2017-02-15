import numpy as np
from scipy.signal import group_delay


def get_filter_delay_at_freq(taps, freq, fs):
    w, h = group_delay((taps, 1))
    w = (w / np.pi) * fs / 2
    return (int(h[np.argmin((w - freq) ** 2)]))
    #width=0.2
    #return (int(np.mean(h[ (w > freq-width) & (w < freq+width)])))

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