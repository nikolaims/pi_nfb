import pylab as plt
from scipy.signal import hilbert

from utils.data.loaders import get_ideal_signal
from utils.sinbase import get_base
from itertools import combinations
import numpy as np

n = 10000
n_components = 50
fs = 250

# load data and sin base
data = get_ideal_signal(causal=True, causal_iir=True, b_order=1, min_phase=True)
signal = data[28:n+28, 15]
ideal = get_ideal_signal()[:n, 15]
n_channels = data.shape[1]
base, freqs = get_base(n, fs=fs, n_components=n_components, low=8, high=12)

# prepare sin
projections = []
pairs = lambda : combinations(range(2*n_components), 2)
for i, j in pairs():
    freq = (1 - 2* (i % 2)) * np.pi/2 * ((i-j)%2) + (freqs[j//2]-freqs[i//2]) * 2 * np.pi * np.arange(n) / fs
    projections.append(2 * np.cos(freq))
projections = np.array(projections + [np.ones((n, )) for k in range(2*n_components)])
c_ind = np.array(list(pairs()) + [(k, k) for k in range(2*n_components)]).T

# rls
from utils.models.rls import DelayedRLSPredictor
rls = DelayedRLSPredictor(2 * n_components, mu=0.6, M=1, delay=0, lambda_=0.999)

prediction = np.zeros_like(signal)
envelope = np.zeros_like(signal)
ws = np.zeros((n, 2 * n_components))
for k in range(n):
    prediction[k] = rls.predict(base[k], signal[k])
    w = rls._w
    ws[k] = w
    envelope[k] = (np.sum(projections[:, k] * w[c_ind[0]] * w[c_ind[1]])) ** 0.5


plt.plot(ws[:, :10], alpha=0.5)
plt.show()

plt.plot(np.abs(ideal), 'b', alpha=0.8)
plt.plot(np.abs(signal), 'g', alpha=0.8)
plt.plot(np.abs(hilbert(signal)), 'g')
plt.plot(np.abs(hilbert(ideal)), 'b')
plt.show()

#plt.plot(np.abs(prediction), 'b', alpha=0.5)

# plt.plot(np.abs(ideal), 'g', alpha=0.5)
plt.plot(np.abs(hilbert(signal)), 'g', linewidth=2, alpha=0.7)
plt.plot(np.abs(hilbert(ideal)), 'k', linewidth=2)

plt.plot(np.abs(hilbert(prediction)), 'm', linewidth=2, alpha=0.7)
plt.plot(envelope, 'r', linewidth=2)

plt.legend(['hilbert fir (-lag 28)', 'hilbert filtfilt', 'hilbert rls_sin_base', 'envelope reconstruction rls_sin_base'])

plt.show()