from scipy.signal import lfilter, hilbert
from utils.data.loaders import get_signal
from utils.filters.ideal_filter import get_fir_filter
import pylab as plt
import numpy as np

from utils.filters.main_freq import get_main_freq
from utils.rls.sparls import sparls
from utils.sinbase import get_base_by_freqs

fs = 250
n = 30000
n_components = 1
raw = get_signal()[:n, 15]
raw = (raw - raw.mean()) / raw.std()

main_freq =  get_main_freq(raw, fs, band=(9, 14), show=1)
i_taps, i_delay = get_fir_filter(fs, main_freq, show=0)
i_signal = lfilter(i_taps, [1.], raw)[i_delay:]
i_envelope = np.abs(hilbert(i_signal))


base = get_base_by_freqs(n, fs, [10, main_freq])
base = base/n_components/2
print(base.shape)
sigma = 1
p, w, w_path = sparls(base, raw, sigma=sigma, alpha=sigma*0.12, lambda_=0.9, K=1, gamma=0., l2=0)


f, [ax1, ax2] = plt.subplots(2)
#ax1.semilogy((p-d_clear)**2)
#ax1.plot(d)
ax1.set_title('Time series')
ax1.set_xlabel('n')

ax2.plot(w_path, alpha=1)
ax2.set_title('Weights path')
ax2.set_ylabel('w')
ax2.set_xlabel('n')

ax1.plot(p)

ax1.plot((w_path**2).sum(1)**0.5/2, 'b', alpha=0.5)
ax1.plot(i_signal, 'g',  alpha=0.5)
#ax1.plot(i_envelope, 'g', alpha=0.5)
print((p[:len(i_signal)] - i_signal).std()/((i_signal).std()))
#print((((w_path**2).sum(1)**0.5/2)[:len(i_signal)] - i_envelope).std()/((i_envelope).std()))

plt.tight_layout()
plt.show()