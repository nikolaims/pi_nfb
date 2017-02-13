from scipy.fftpack import rfft, irfft, fftfreq

def fft_filter(signal, band, fs):
    W = fftfreq(signal.size, d=1/fs*2)
    f_signal = rfft(signal)
    cut_f_signal = f_signal.copy()
    cut_f_signal[(W<band[0]) | (W>band[1])] = 0
    cut_signal = irfft(cut_f_signal)
    return cut_signal