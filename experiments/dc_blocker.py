from utils.data.loaders import load_feedback
from utils.metrics import truncated_nmse
import pylab as plt
import numpy as np
from scipy.signal import *
from utils.filters import dc_blocker

data, signal, derived = load_feedback(ica_artifact=True)
fs = 250
plt.plot(np.arange(signal.shape[0])/fs, signal)
plt.show()



# DC Blocker https://ccrma.stanford.edu/~jos/fp/DC_Blocker.html
y = dc_blocker(signal)

b, a = butter(4, 0.5/152, btype='high')
plt.plot(y)
plt.plot(filtfilt(b, a, signal),'.-')
plt.show()


b, a = butter(4, [8/125, 12/125], btype='band')
plt.plot(filtfilt(b, a, signal),'.-')
plt.plot(filtfilt(b, a, y),'.-')

plt.show()