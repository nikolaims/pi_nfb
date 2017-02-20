from scipy.signal import firwin2

from utils.filters import minimum_phase


def get_fir_filter(fs, main_freq=10, order=2000, width=1, show=False, band=None):
    # design ideal filter
    if band is None:
        band = (main_freq - width, main_freq + width)
    else:
        print('Warning!! band was defined! Main_freq and width will be ignored!! ')

    print(band)
    w = 1
    freq = [0, band[0]-w, band[0], band[1], band[1]+w, fs/2]
    gain = [0, 0.1, 1, 0.8, 0, 0]
    taps = firwin2(order, freq, [0, 0, 1, 1, 0, 0], nyq=fs/2)
    if show:
        from mne1.viz import plot_filter
        plot_filter(taps, fs, freq, gain, flim=(5, 15), fscale='linear')
    return taps, order//2

def get_fir_filter_high_pass(fs, main_freq=10, order=2000, width=1, show=False):
    # design ideal filter
    w = 0
    freq = [0, main_freq-w, main_freq, fs/2]
    gain = [0, 0, 1, 1]
    taps = firwin2(order, freq, gain, nyq=fs/2)
    if show:
        from mne1.viz import plot_filter
        plot_filter(taps, fs, freq, gain, flim=(5, 15), fscale='linear')
    return taps, order // 2


if __name__ == '__main__':
    #get_fir_filter(250, show=True)
    taps = get_fir_filter_high_pass(250, main_freq=10-1, order=201, width=1, show=1)
    min_phase_taps = minimum_phase(taps)
    from mne1.viz import plot_filter
    plot_filter(min_phase_taps, 250, fscale='linear', flim=(0, 20))
