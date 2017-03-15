import numpy as np
import pylab as plt
from utils.data.loaders import load_normalised_raw_signal
from utils.envelope.smoothers import exp_smooth
from utils.filters.fft_chunk import fft_chunk_envelope
from utils.filters.main_freq import get_main_freq
from utils.pipeline.ideal_envelope_detector import ideal_envelope_detector
from utils.pipeline.naive_rls import sparls_ed
from utils.sinbase import get_base_by_freqs
from utils.metrics import lag_compensed_nmse, find_lag, smoothness
nor = lambda x: (x - x.mean()) / x.std()

fs = 250
n = 20000
t = np.arange(n)/fs

# load signal
raw = load_normalised_raw_signal()

# get main freq
main_freq = get_main_freq(raw, fs, band=(8, 12))
band = (main_freq - 1, main_freq + 1)

# ideal signal
i_signal, i_envelope = ideal_envelope_detector(raw, band, fs, n)

# truncate samples
raw = raw[:n]

# naive rls
freqs = [main_freq]
a, h = np.linspace(main_freq - 1, main_freq, 20, endpoint=False, retstep=True)
freqs = list(a) + [main_freq] + list(a + 1 + h)

base = get_base_by_freqs(n, fs, freqs)/(len(freqs)*2)

mus = [1.2]# np.linspace(1, 1.5, 10)
mses = np.zeros_like(mus)
lags = np.zeros_like(mus)
for k, mu in []:#enumerate(mus):
    print(k)
    signal, envelope = sparls_ed(raw, band, fs, freqs, alpha=mu, lambda_=0.9, K=1, gamma=0.01, l2=0.001)
    lags[k], mses[k] = lag_compensed_nmse(envelope, i_envelope, show=0)


opt_lag = 20
plt.plot(mus, lags/opt_lag)
plt.plot(mus, mses)
plt.plot(mus, lags/opt_lag + mses)
opt_mu = mus[np.argmin(lags/opt_lag + mses)]
print(opt_mu)
plt.legend(['lag/100ms', 'nmse', 'mixed'])

plt.show()

signal, envelope = sparls_ed(raw, band, fs, freqs, alpha=opt_mu, lambda_=0.9, K=1, gamma=1, l2=0.001)
envelope = exp_smooth(envelope, factor=0.05)
find_lag(envelope, i_envelope, fs, 1)
plt.show()



