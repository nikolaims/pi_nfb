import numpy as np
import pylab as plt

indexes = np.arange(2500)


fs = 250
a = indexes/indexes.max()
b = np.sin(2*np.pi* indexes/fs )
signal = a * np.cos(10*2*np.pi* indexes/fs) + b * np.sin(11 * 2*np.pi * indexes/fs) + b * np.sin(12 * 2*np.pi * indexes/fs)
plt.plot(indexes/fs, signal)

plt.plot(indexes/fs, ((a)**2+(b)**2 +(b)**2 + 2*a*b*np.cos(-np.pi / 2 + 2*np.pi* indexes/fs) + 2*a*b*np.cos(-np.pi / 2 + 2*2*np.pi* indexes/fs)
                      + 2 * b * b * np.cos(-1* 2 * np.pi * indexes / fs))**0.5)


plt.show()