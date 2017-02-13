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
b, a = butter(4, [8/fs*2, 12/fs*2], 'band')

signal_filtfilt = filtfilt(b, a, signal)
plt.plot(signal, alpha=0.1)


# time
t = np.arange(signal.shape[0]) / fs

# model
n_components = 100
base = []
for model_freq in np.linspace(8-0, 12+0, n_components):
    base += [np.sin(model_freq * t * 2 * np.pi), np.cos(model_freq * t * 2 * np.pi)]
base = np.vstack(base).T

# magic filter
from utils.filters import magic_filter_taps
signal = lfilter(magic_filter_taps(), 1.0, signal, axis=0)
delay = 14
plt.plot(signal[delay:])
plt.plot(signal_filtfilt)
plt.show()

prediction = rls_predict_many(base, signal, M=3, lambda_=0.9995, delta_var=1000, mu=1)
print(truncated_nmse(prediction[delay:], signal_filtfilt[:-delay], start_from=stop_on))

# plot
plt.plot(signal[delay:])
plt.plot(prediction[delay:])
#print(signal_filtfilt[delay:].shape, prediction[:-delay].shape, t[:-delay].shape)
plt.plot(signal_filtfilt)
plt.plot(prediction[delay:] - signal_filtfilt[:-delay], alpha=0.9)
plt.legend(['raw', 'prediction', 'filtfilt', 'error'])
plt.show()