from scipy.signal import firwin2



def get_fir_filter(fs, main_freq=10, order=2000, width=1, show=False):
    # design ideal filter
    band = (main_freq - width, main_freq + width)
    w = 0.1
    freq = [0, band[0]-w, band[0], band[1], band[1]+w, fs/2]
    gain = [0, 0, 1, 1, 0, 0]
    taps = firwin2(order, freq, [0, 0, 1, 1, 0, 0], nyq=fs/2)
    if show:
        from mne1.viz import plot_filter
        plot_filter(taps, fs, freq, gain, flim=(5, 15), fscale='linear')
    return taps, order//2

if __name__ == '__main__':
    get_fir_filter(250, show=True)