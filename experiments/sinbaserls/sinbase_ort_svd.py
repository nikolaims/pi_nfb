from numpy.linalg import svd
from utils.sinbase import get_base_ort
import pylab as plt
import numpy as np

base, freqs = get_base_ort(500, 250,low=9, high=11, n_components=20)
print(freqs, len(freqs))
print(np.dot(base[:, 0], base[:, 1]))

#plt.plot(base)
#plt.show()

u, s, v = svd(base, full_matrices=False)
print([x.shape for x in [base, u, s, v]])
print(np.dot(u[:, 0], u[:, 0]))

ort_base = np.dot(base, v.T)
plt.plot(u)
plt.show()


w = np.random.normal(size=(20,))
plt.plot(np.sum(ort_base, axis=1))
plt.plot(np.sum(base, axis=1))
plt.show()
