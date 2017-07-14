import numpy as np
from scipy.fftpack import fftfreq, rfft

def fft_chunk_envelope(raw, band, fs, smoothing_factor=0.3, chunk_size=8, n_samples=500):
    # fft chunk filter window
    # asymmetric gaussian window
    p = round(2 * n_samples * 2 / 4)  # maximum
    eps = 0.0001  # bounds value
    power = 2  # power of x
    left_c = - np.log(eps) / (p ** power)
    right_c = - np.log(eps) / (2 * n_samples - 1 - p) ** power

    samples_window = np.concatenate([np.exp(-left_c * abs(np.arange(p) - p) ** power),
                                     np.exp(-right_c * abs(np.arange(p, 2 * n_samples) - p) ** power)])

    # helpers
    w = fftfreq(2 * n_samples, d=1. / fs * 2)

    # filter
    filtered = np.zeros_like(raw)
    smoothed = np.zeros_like(raw)

    previous_sample = 0
    for k in range(n_samples, len(filtered), chunk_size):
        buffer = raw[k-n_samples:k]
        f_signal = rfft(np.hstack((buffer, buffer[-1::-1])) * samples_window)
        cut_f_signal = f_signal.copy()
        cut_f_signal[(w < band[0]) | (w > band[1])] = 0  # TODO: in one row
        filtered_sample = np.abs(cut_f_signal).mean() / np.mean(samples_window)/2
        filtered[k - chunk_size + 1:k + 1] = filtered_sample
        current_sample = smoothing_factor * filtered_sample + (1 - smoothing_factor) * previous_sample
        smoothed[k-chunk_size+1:k+1] = current_sample
        previous_sample = current_sample
    return smoothed#, filtered, samples_window