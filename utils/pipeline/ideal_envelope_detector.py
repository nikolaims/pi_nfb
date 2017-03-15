import numpy as np
from scipy.signal import lfilter, hilbert
from utils.filters.ideal_filter import get_fir_filter

def ideal_envelope_detector(raw, band, fs, n=None):
    i_taps, i_delay = get_fir_filter(fs, None, show=0, band=band)
    i_signal = lfilter(i_taps, [1.], raw)[i_delay:][:n]
    i_envelope = np.abs(hilbert(i_signal))
    return i_signal, i_envelope