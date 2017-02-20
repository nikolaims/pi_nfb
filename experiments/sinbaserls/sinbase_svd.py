from numpy.linalg import svd
from utils.sinbase import get_base
import pylab as plt
import numpy as np

base, freqs = get_base(20000, 250, n_components=10, low=9, high=11)

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
