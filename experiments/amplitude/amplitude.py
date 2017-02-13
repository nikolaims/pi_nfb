import numpy as np
import pylab as plt
from utils.sinbase import get_base

n = 250*10
fs = 250
n_components = 50
base, freqs = get_base(n, fs, n_components, low=8, high=12)
w = np.random.normal(size=(2*n_components, n))
from scipy.signal import butter, filtfilt
b, a = butter(4, 5/fs*2)

w = filtfilt(b, a, w, 1)
plt.plot(w.T)



signal = np.sum(base.T* w, 0)
plt.show()
print(signal.shape)

from itertools import combinations

projections = []
pairs = lambda : combinations(range(2*n_components), 2)
for i, j in pairs():
    # print(i, j, (1 - 2* (i % 2)) * ((i-j)%2), freqs[i//2]-freqs[j//2])
    freq = (1 - 2* (i % 2)) * np.pi/2 * ((i-j)%2) + (freqs[j//2]-freqs[i//2]) * 2 * np.pi * np.arange(n) / fs
    projections.append(2 * np.cos(freq))

plt.plot(np.array(projections[:100]).T + np.arange(100)*2)
plt.show()

print(w[0].shape)


projections_pairs = np.array([w[i] * w[j] for k, (i, j) in enumerate(pairs())])
projections = np.array(projections + [np.ones((n, )) for k in range(2*n_components)])
print(projections.shape, projections_pairs.shape)
ampl = np.zeros((n, ))
timer = np.zeros((n, ))
from time import time


inds = np.array(list(pairs()) + [(k, k) for k in range(2*n_components)]).T
print(inds)
for k in range(n):
    wk = w[:, k]

    t = time()
    ampl[k] = (np.sum(projections[:, k] * wk[inds[0]] * wk[inds[1]]))**0.5
    timer[k] = time() - t
print(np.median(timer), timer.max())
plt.hist(timer, bins=100)
plt.show()

plt.plot(signal)
plt.plot(ampl)
plt.plot(-ampl)
plt.show()