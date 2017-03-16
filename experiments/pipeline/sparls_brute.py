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

def f(alpha, lambda_, gamma, l2):
    try:
        signal, envelope = sparls_ed(raw, band, fs, freqs, alpha=alpha, lambda_=lambda_, K=1, gamma=gamma, l2=l2)
    except Exception:
        return None, None, None
    lag, mse = lag_compensed_nmse(envelope, i_envelope, show=0)
    return lag, mse, lag/25 + mse

alpha_range = np.linspace(0.7, 1.3, 5)
lambda_range = np.linspace(0.7, 0.999,5)
gamma_range = [0., 0.001, 0.01]
l2_range = [0., 0.001, 0.01]


import pandas as pd
ds = pd.DataFrame(columns=['f', 'lag', 'nmse', 'alpha', 'lambda', 'gamma', 'l2'])
k = 0
for alpha in alpha_range:
    for lambda_ in lambda_range:
        for gamma in gamma_range:
            for l2 in l2_range:
                lag, mse, fv = f(alpha, lambda_, gamma, l2)
                #print('{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t'.format(alpha, lambda_, lag, mse, fv))
                ds.loc[k] = [fv,  lag,  mse,  alpha, lambda_, gamma, l2]
                print(ds.loc[k:k+1])
                k += 1
print(ds)
from time import time
ds.to_csv('res{}.csv'.format(round(time())))
