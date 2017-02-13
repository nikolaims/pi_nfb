from ftplib import error_proto

from utils.data.loaders import load_feedback
from utils.metrics import truncated_nmse
import pylab as plt
import numpy as np
from scipy.signal import *
from utils.filters import dc_blocker
from utils.rls.rls import rls_predict_many, rls_predict
from utils.metrics import truncated_nmse
import seaborn
cm = seaborn.color_palette()

data, signal, derived = load_feedback(ica_artifact=False)
fs = 250
nn = 5000
slc = range(10000, 15000)
slc2 = range(10000, 11000)

plt.figure(figsize=(5,3))
plt.plot(slc, signal[slc])
plt.savefig('raw.png', dpi=200)
plt.show()

# dc filter
signal = dc_blocker(signal)
data = dc_blocker(data)

plt.figure(figsize=(5,3))
plt.plot(slc, signal[slc])
plt.savefig('raw_dc.png', dpi=200)
plt.show()


data, signal, derived = load_feedback(ica_artifact=True)
# dc filter
signal = dc_blocker(signal)
data = dc_blocker(data)

plt.figure(figsize=(5,3))
plt.plot(slc, signal[slc])
plt.savefig('raw_dc_ica.png', dpi=200)
plt.show()

# normalizing
stop_on = 1000
signal = (signal - signal[stop_on:].mean()) / signal[stop_on:].std()
data = (data - data[stop_on:].mean(0)) / data[stop_on:].std(0)
plt.figure(figsize=(5,3))
plt.plot(slc, signal[slc])
plt.savefig('raw_dc_ica_norm.png', dpi=200)
plt.show()


# filtfilt signal
b, a = butter(5, [7/fs*2, 13/fs*2], 'band')
signal_filtfilt = filtfilt(b, a, signal)


# plot raw and filtfilt
plt.figure(figsize=(5,3))
plt.plot(slc2, signal[slc2], alpha=0.3)
plt.plot(slc2, signal_filtfilt[slc2], c='g')
plt.legend(['raw', 'filtfilt'])
plt.savefig('raw_filtfilt.png', dpi=200)
plt.show()

# magic filter
from utils.filters import magic_filter_taps
signal = lfilter(magic_filter_taps(bad=True), 1.0, signal, axis=0)
delay = 14
plt.figure(figsize=(5,3))
plt.plot(slc2, signal[slc2])
plt.plot(slc2, signal_filtfilt[slc2], alpha=0.8, c='g')
plt.legend(['filter', 'filtfilt'])
plt.savefig('filter_filtfilt.png', dpi=200)
plt.show()

# time
t = np.arange(signal.shape[0]) / fs

# model
n_components = 50
base = []
for model_freq in np.linspace(9, 14, n_components):
    base += [np.sin(model_freq * t * 2 * np.pi), np.cos(model_freq * t * 2 * np.pi)]
base = np.vstack(base).T

# fitting
prediction = rls_predict_many(base, signal, M=1, lambda_=0.9995, delta_var=1000, mu=1)
print(truncated_nmse(prediction[delay:], signal_filtfilt[:-delay], start_from=stop_on))

# plot
plt.figure(figsize=(10,4))

plt.plot(slc2, (prediction[delay:] - signal_filtfilt[:-delay])[slc2], c=cm[4])
plt.plot(slc2,signal[delay:][slc2], alpha=0.5)
#print(signal_filtfilt[delay:].shape, prediction[:-delay].shape, t[:-delay].shape)
plt.plot(slc2,signal_filtfilt[slc2], 'g')

plt.plot(slc2,prediction[delay:][slc2], c='r')
plt.legend(['filt', 'filtfilt', 'prediction', ])

plt.savefig('prediction.png', dpi=200)
plt.show()