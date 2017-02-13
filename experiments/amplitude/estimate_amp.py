import pylab as plt
from utils.data.loaders import load_feedback
from utils.filters import dc_blocker
import numpy as np

data, _signal, _derived = load_feedback(ica_artifact=True)
data = dc_blocker(data)
fs = 250
nq = 250/2

from scipy.signal import butter, lfilter, filtfilt, hilbert, firwin
from utils.filters import magic_filter_taps, custom_fir
b, a = butter(4, [8/nq, 12/nq], btype='band')

x = filtfilt(b, a, data[:, 15])

amplitude_envelope = np.abs(hilbert(x))

def simple_envlope(x, factor=0.95):
    x = np.abs(x)
    y = np.zeros_like(x)
    for k in range(1, len(x)):
        if x[k] > y[k-1]:
            y[k] = x[k]
        else:
            y[k] = y[k-1] * factor + x[k] * (1 - factor)
    return y

def peak_envlope_lfilter(x, factor=0.95, fir=False):
    x = np.abs(x)
    if fir:
        a = 1
        b = firwin(20, 3/nq)
    else:
        b, a = butter(1, 3/nq, btype='low')
    y = lfilter(b, a, x)
    y *= np.std(x)/np.std(y)*1.1
    return y




plt.plot(x, alpha=0.2)
plt.plot(amplitude_envelope)
plt.plot(simple_envlope(x))
plt.plot(peak_envlope_lfilter(x, fir=True))
plt.plot(peak_envlope_lfilter(x, fir=False))

plt.legend(['signal', 'hilbert', 'diode-rc', 'fir', 'irr'])
plt.show()


