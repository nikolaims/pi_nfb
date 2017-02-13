import numpy as np
import pylab as plt
from scipy.signal import *

# nyquist rate
nq = 250 / 2

# transition width
width = 7/nq

# attenuation in the stop band [db]
ripple_db = 30.0

# compute the order and Kaiser parameter for the FIR filter.
N, beta = kaiserord(ripple_db, width)
print('FIR filter order:', N)

# create a bandpass FIR filter
taps = firwin(N, [7/nq, 13/nq], window=('kaiser', beta), pass_zero=False)
print(sum(taps))

# plot filter magnitude response
p = 0.3
ma = np.array([p**k for k in range(5)])
ma = ma / ma.mean()
w, h = freqz(ma, worN=8000)
plt.plot((w/np.pi)*nq, np.absolute(h), linewidth=2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.title('Frequency Response')
plt.ylim(-0.05, 1.05)
plt.xlim(0, 30)
plt.vlines([8, 12], [-0.05, -0.05], [1.05, 1.05])
plt.xticks(np.arange(0, 31, 2))
plt.show()