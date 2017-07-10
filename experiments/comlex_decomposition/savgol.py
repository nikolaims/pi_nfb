from utils.data.loaders import load_normalised_raw_signal
import numpy as np
import pylab as plt
from scipy.signal import welch, lfilter, savgol_coeffs, butter, minimum_phase

from utils.filters.fft_chunk import fft_chunk_envelope
from utils.filters.main_freq import get_main_freq
from utils.metrics import find_lag
from utils.pipeline.ideal_envelope_detector import ideal_envelope_detector
nor = lambda x: (x - x.mean()) / x.std()


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


sine = np.exp(-1j*np.arange(25000)/fs*2*np.pi*(main_freq))
b, a = butter(1, 1.5/fs*2, )
iir_filtered = lfilter(b, a, raw * sine)
n_taps, ordd = 151, 2
sc = savgol_coeffs(n_taps, ordd, pos=n_taps-1)
am3 = lfilter(sc, [1.], iir_filtered)
x_savgol = np.abs(2 * am3)

# butter + rc filter
b, a = butter(1, np.array(band)/fs*2, btype='band')
x_butter = lfilter(b, a, raw)
from utils.envelope.smoothers import exp_smooth
x_butter = exp_smooth(np.abs(x_butter), 0.025)

# fft
x_fft = fft_chunk_envelope(raw, band=(main_freq - fn, main_freq + fn), fs=fs, smoothing_factor=0.1, chunk_size=1)

def find_lag0(target, x, n=200):
    nor = lambda x: (x - np.mean(x)) / np.std(x)
    lags = np.arange(n)
    mses = np.zeros_like(lags).astype(float)
    n_points = len(target) - n
    for lag in lags:
        mses[lag] = np.mean((nor(target[:n_points]) * nor(x[lag:n_points + lag])))
    lag = np.argmax(mses)
    return lag, mses[lag]

# collect statistics
n_windows = 30
n_samples = n // n_windows
envelopes_names = ['STFT', 'BE', 'CD-SG']
envelopes = dict(zip(envelopes_names, [x_fft,  x_butter, x_savgol]))
import pandas as pd
from utils.metrics import smoothness
stats = pd.DataFrame(columns=['method', 'correlation', 'smoothness', 'lag, ms'])
for k in range(n_windows):
    window_slice = slice(n_samples*k, n_samples*(k+1))
    for envelope in envelopes_names:
        x = nor(envelopes[envelope])
        lag, corr = find_lag0(i_envelope[window_slice], x[window_slice], n=200)
        smooth = smoothness(x[window_slice], nor(i_envelope)[window_slice])
        stats.loc[len(stats)+1] = [envelope, corr, smooth, lag/fs*1000]
print(stats)
import seaborn as sns
f = plt.figure()
axes = [f.add_subplot(2, 3, k) for k in range(4, 7)]
axes[0].set_title('b')
axes[1].set_title('c')
axes[2].set_title('d')
sns.boxplot(x="method", y="lag, ms", data=stats, ax=axes[0])
sns.boxplot(x="method", y="correlation", data=stats, ax=axes[1])
sns.boxplot(x="method", y="smoothness", data=stats, ax=axes[2])

ax = f.add_subplot(2, 1, 1)
ax.set_title('a')
cm = sns.color_palette()
t = np.arange(len(i_signal))/fs
ax.plot(t, i_signal**2, c=cm[4], alpha=0.5)
ax.plot(t, i_envelope**2, c=cm[4], alpha=1)
ax.plot(t, x_fft ** 2, c=cm[0], alpha=1)
ax.plot(t, x_savgol ** 2 / np.std(x_savgol ** 2) * np.std(i_envelope ** 2), c=cm[2], alpha=1)
ax.plot(t, x_butter ** 2/np.std(x_butter ** 2)*np.std(i_envelope**2), c=cm[1], alpha=1)
ax.legend(['$X^2(n)$','$P(n)$','$P_{STFT}(n)$','$P_{CD-SG}(n)$', '$P_{BE}(n)$'])
ax.set_xlabel('time, s')
ax.set_ylabel('$P$')
ax.set_xlim(8600/fs, 9800/fs)
plt.tight_layout()
plt.show()

if 1:
    import seaborn as sns
    sns.set_style("white")
    f = plt.figure(figsize=(7, 5))
    ax = f.add_subplot(2, 1, 2)
    ax.set_title('b')
    nor = lambda x: (x - np.mean(x) * 0) / np.std(x)


    cm = sns.color_palette()
    plt.plot(i_signal**2, c=cm[0], alpha=0.5)
    plt.plot(i_envelope**2, c=cm[0], alpha=1)
    plt.plot(x_fft ** 2, c=cm[1], alpha=1)
    plt.plot(x_savgol ** 2 / np.std(x_savgol ** 2) * np.std(i_envelope ** 2), c=cm[2], alpha=1)
    plt.plot(x_butter ** 2/np.std(x_butter ** 2)*np.std(i_envelope**2), c=cm[4], alpha=1)
    plt.legend(['$X^2(n)$','$P(n)$','$P_{FFT}(n)$','$P_{CD-SG}(n)$', '$P_{BAE}(n)$'])
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
    for j, data in enumerate([x_fft, x_savgol, x_butter]):
        c = cm[[1, 2, 4][j]]
        lag, mses = find_lag0(data)
        hdls.append(plt.plot(mses, c=c, label=['CD-SG', 'FFT', 'BAE'][j])[0])
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

