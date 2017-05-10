from utils.data.loaders import load_normalised_raw_signal
import numpy as np
import pylab as plt
from scipy.signal import welch, lfilter, savgol_coeffs, butter, minimum_phase

from utils.filters.fft_chunk import fft_chunk_envelope
from utils.filters.main_freq import get_main_freq
from utils.metrics import find_lag
from utils.pipeline.ideal_envelope_detector import ideal_envelope_detector
nor = lambda x: (x - x.mean()) / x.std()


def plot_filters(b, a, sc, fs, fn):
    from mne1.viz import plot_filter
    f = plot_filter({'b': b, 'a': a}, fs, show=False)
    plot_filter(sc, fs, fig_axes=f, color='g', fscale='linear', freq=[0, fn, fn, fs/2], gain=[1, 1, 0, 0], flim=(0,10),
                show=False)
    plt.legend(['butter', 'savgol', 'ideal'])
    plt.show()


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

# truncate samples
raw = raw[:n]


#plt.plot(raw)
#plt.plot(i_signal)
#plt.plot(i_envelope)
#plt.show()

sine = np.exp(-1j*np.arange(25000)/fs*2*np.pi*(main_freq))



b, a = butter(1, 1.5/fs*2, )
iir_filtered = lfilter(b, a, raw * sine)
n_taps, ordd = 151, 2



#n_taps, ordd= 209-20, 2
#n_taps, ordd= 517, 3
sc = savgol_coeffs(n_taps, ordd, pos=n_taps-1)
print(max(np.abs(np.roots(sc))))
#sc = minimum_phase(sc)
print(max(np.abs(np.roots(sc))))
print(sc)

am3 = lfilter(sc, [1.], iir_filtered)
x = np.abs(2 * am3)

find_lag(x, i_envelope, fs, True)


# butter + rc filter
b, a = butter(1, np.array(band)/fs*2, btype='band')
x_butter = lfilter(b, a, raw)
from utils.envelope import RCEnvelopeDetector
from utils.envelope.smoothers import exp_smooth
#env = RCEnvelopeDetector(0.8)
#x_butter = np.array([env.get_envelope(x_) for x_ in x])
x_butter = exp_smooth(np.abs(x_butter), 0.025)
find_lag(x_butter, i_envelope, fs, True)

if 1:
    import seaborn as sns
    sns.set_style("white")
    f = plt.figure(figsize=(7, 5))
    ax = f.add_subplot(2, 1, 2)
    ax.set_title('b')
    nor = lambda x: (x - np.mean(x) * 0) / np.std(x)

    fft_envelope = fft_chunk_envelope(raw, band=(main_freq-fn, main_freq+fn), fs=fs, smoothing_factor=0.1, chunk_size=1)
    cm = sns.color_palette()
    plt.plot(i_signal**2, c=cm[0], alpha=0.5)
    plt.plot(i_envelope**2, c=cm[0], alpha=1)
    plt.plot(fft_envelope**2, c=cm[1], alpha=1)
    plt.plot(x ** 2/np.std(x ** 2)*np.std(i_envelope**2), c=cm[2], alpha=1)
    plt.plot(x_butter ** 2/np.std(x_butter ** 2)*np.std(i_envelope**2), c=cm[4], alpha=1)
    plt.legend(['$X^2(n)$','$P(n)$','$P_{FFT}(n)$','$P_{CM-SG}(n)$', '$P_{BAE}(n)$'])
    plt.xlabel('$n$, [samples]')
    plt.ylabel('$P$')
    plt.xlim(8600, 9800)

    def find_lag0(x):
        n = 10000
        target = i_envelope
        nor = lambda x:  (x - np.mean(x)) / np.std(x)
        lags = np.arange(n)
        mses = np.zeros_like(lags).astype(float)
        n_points = len(target) - n
        for lag in lags:
            mses[lag] = np.mean((nor(target[:n_points]) * nor(x[lag:n_points+lag])))
        lag = np.argmax(mses)
        return lag, mses


    ax1 = f.add_subplot(2, 1, 1)

    ax1.set_title('a')
    hdls = []
    for j, data in enumerate([fft_envelope, x, x_butter]):
        c = cm[[1, 2, 4][j]]
        lag, mses = find_lag0(data)
        hdls.append(plt.plot(mses, c=c, label=['CM-SG', 'FFT', 'BAE'][j])[0])
        plt.plot(lag, np.max(mses), 'o', c=c)
        lag_str = '{}'.format(lag) if fs is None else '{} ({:.3f} s)'.format(lag, lag/fs)
        plt.text(lag-10, np.max(mses)+0.1*j+0.1, lag_str, color=c)

    plt.ylim(0, 1.5)
    plt.xlim(0, 150)
    plt.xlabel('lag, [samples]')
    plt.ylabel('corr')
    plt.legend(hdls, ['CM-SG', 'FFT', 'BAE'])
    f.tight_layout()
    f.savefig('conf.jpg', dpi=300)

    plt.show()

