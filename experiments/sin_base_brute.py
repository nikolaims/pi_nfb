from utils.data.loaders import load_feedback
from utils.metrics import truncated_nmse
import pylab as plt
import numpy as np
from scipy.signal import *
from utils.filters import dc_blocker
from utils.rls.rls import rls_predict_many, rls_predict
from utils.metrics import truncated_nmse

data, signal, derived = load_feedback(ica_artifact=True)
fs = 250

# dc filter
signal = dc_blocker(signal)
data = dc_blocker(data)

# normalizing
stop_on = 1000
signal = (signal - signal[stop_on:].mean()) / signal[stop_on:].std()
data = (data - data[stop_on:].mean(0)) / data[stop_on:].std(0)

# filtfilt signal
b, a = butter(5, [9/fs*2, 14/fs*2], 'band')
signal_filtfilt = filtfilt(b, a, signal)

# time
t = np.arange(signal.shape[0]) / fs

# brute
def f(n_components, M, lambda_, mu):
    M = int(M)
    n_components = int(n_components)
    # model
    base = []
    for model_freq in np.linspace(9, 14, n_components):
        base += [np.sin(model_freq * t * 2 * np.pi), np.cos(model_freq * t * 2 * np.pi)]
    base = np.vstack(base).T

    # fitting
    prediction = rls_predict_many(base, signal, M=M, lambda_=lambda_, delta_var=100, mu=mu)
    return truncated_nmse(prediction, signal_filtfilt, start_from=stop_on)

res = []
best = 1000000
best_ind = 0
ind = 0
for n_components in range(1, 21, 4):
    for M in range(1, 15, 4):
        for lambda_ in np.linspace(0.999, 1, 3):
            for mu in np.linspace(0.5, 1, 3):
                nmse = f(n_components, M, lambda_, mu)
                if nmse < best:
                    best = nmse
                    best_ind = ind
                res.append((nmse, n_components, M, lambda_, mu))
                ind += 1
                print('{}\tbest: {}\tn_components: {}\tM: {}\tlambda: {}\tmu: {}\t'.format(nmse, best, n_components, M,
                                                                                           lambda_, mu))

from  pickle import dump
dump((best, best_ind, res), open( "save.p", "wb"))

