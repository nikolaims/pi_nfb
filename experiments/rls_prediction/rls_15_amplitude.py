import pylab as plt
from utils.data.loaders import load_feedback
from utils.filters import dc_blocker
import numpy as np

from utils.rls.rls import DelayedRLSPredictor

data, _signal, _derived = load_feedback(ica_artifact=True)
data = dc_blocker(data)
fs = 250
nq = 250/2

#plt.plot(data + np.arange(data.shape[1])*100)
#plt.show()

# filter data
from scipy.signal import butter, lfilter, filtfilt, hilbert
from utils.filters import magic_filter_taps, custom_fir
b, a = butter(4, [8/nq, 12/nq], btype='band')

signal_filtfilt = filtfilt(b, a, data[:, 15])

taps, delay = custom_fir()
data_filt = lfilter(taps, 1.0, data, axis=0)
signal_filt = data_filt[:, 15]

plt.plot(signal_filt[delay:], alpha=1)
plt.plot(signal_filtfilt)
plt.show()

x = signal_filt[delay:, None]
#plt.plot(x)
x = np.abs(hilbert(x.flatten()))
data_h = np.abs(np.array([hilbert(data_filt[delay:, j]) for j in range(data_filt.shape[1])]).T)
data_h = (data_h - data_h.mean(0)) / data_h.std(0)
plt.plot(data_h)
plt.show()
x = (x - np.mean(x)) / np.std(x)
plt.plot(x)
plt.show()
print(x.shape)
x_pred = np.zeros_like(x)
subdelay = 0
rls = DelayedRLSPredictor(n_channels=data_h.shape[1], delay=delay+subdelay, target_channel=15, M=10, lambda_=0.9999, mu=1)
for k in range(len(x) - delay - subdelay):
    x_pred[k + delay] = rls.predict(data_h[subdelay:][k])

plt.plot(x)
plt.plot(x_pred)
plt.legend(['x', 'x_pred'])
plt.ylim(-100, 100)
plt.show()



plt.plot(np.abs(hilbert(signal_filt[delay:])), alpha=1)
plt.plot(np.abs(hilbert(signal_filtfilt)))
plt.plot(np.abs(hilbert(x_pred)))
plt.show()



def simple_envlope(x, factor=0.95):
    x = np.abs(x)
    y = np.zeros_like(x)
    for k in range(1, len(x)):
        if x[k] > y[k-1]:
            y[k] = x[k]
        else:
            y[k] = y[k-1] * factor + x[k] * (1 - factor)
    return y

plt.plot(simple_envlope((signal_filt[delay:])), alpha=1)
plt.plot(np.abs(hilbert(signal_filtfilt)))
#plt.plot(simple_envlope((x_pred)))
plt.show()