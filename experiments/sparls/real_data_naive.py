from scipy.signal import lfilter, hilbert
from utils.data.loaders import get_signal
from utils.envelope import find_lag
from utils.filters import dc_blocker
from utils.filters.ideal_filter import get_fir_filter
import pylab as plt
import numpy as np

from utils.filters.main_freq import get_main_freq
from utils.rls.naive import naive_rls
from utils.rls.sparls import sparls
from utils.sinbase import get_base_by_freqs, get_base

fs = 250
n = 30000
n_components = 1
raw = get_signal()[:n, 15]
raw = (raw - raw.mean()) / raw.std()

main_freq =  get_main_freq(raw, fs, band=(9, 14), show=0)
i_taps, i_delay = get_fir_filter(fs, main_freq, show=0)
i_signal = lfilter(i_taps, [1.], raw)[i_delay:]
i_envelope = np.abs(hilbert(i_signal))


base = get_base_by_freqs(n, fs, [main_freq])
ww=0.9
#base, fss = get_base(n, fs=fs, n_components=n_components, low=main_freq - ww, high=main_freq + ww, nonlinear=1.25)
base = base/n_components/2
print(base.shape)
sigma = 1
#p, w, w_path = sparls(base, raw, sigma=sigma, alpha=sigma*0.12, lambda_=0.9, K=1, gamma=0., l2=0)
p, w, w_path = naive_rls(base, raw, mu=0.15)


f, [ax1, ax2] = plt.subplots(2)
#ax1.semilogy((p-d_clear)**2)
#ax1.plot(d)
ax1.set_title('Time series')
ax1.set_xlabel('n')

ax2.plot(w_path, alpha=1)
ax2.set_title('Weights path')
ax2.set_ylabel('w')
ax2.set_xlabel('n')


ax1.plot(p**2)


env = (w_path**2).sum(1)**0.5/2

ax1.plot(env**2, 'b', alpha=0.5)
ax1.plot(i_signal**2, 'g',  alpha=0.9)
ax1.plot(i_envelope**2, 'g',  alpha=0.9)
#ax1.plot(i_envelope, 'g', alpha=0.5)
print((p[:len(i_signal)] - i_signal).std()/((i_signal).std()))
#print((((w_path**2).sum(1)**0.5/2)[:len(i_signal)] - i_envelope).std()/((i_envelope).std()))
ax2.set_xlim(18000, 20000)
ax1.set_xlim(18000, 20000)
plt.tight_layout()
plt.show()


p1, w1, w_path1 = naive_rls(base, env, mu=0.5)

y = np.zeros_like(env)
for k in range(1, len(y)):
    y[k] = y[k-1]*0.8 + (env-p1)[k]*0.2
find_lag(y, i_envelope, 250, show=True)

from scipy.signal import welch
plt.figure()
plt.plot(*welch(env-p1, 250, nperseg=1000))
plt.show()



p1, w1, w_path1 = naive_rls(base, w_path[:, 0], mu=0.5)
w_path[:, 0] -= p1
p1, w1, w_path1 = naive_rls(base, w_path[:, 1], mu=0.5)
w_path[:, 1] -= p1

y = np.zeros_like(env)
for k in range(1, len(y)):
    y[k] = y[k-1]*0.8 + (env-p1)[k]*0.2
find_lag(y, i_envelope, 250, show=True)

from scipy.signal import welch
plt.figure()
f, [ax1, ax2] = plt.subplots(2)
#ax1.semilogy((p-d_clear)**2)
#ax1.plot(d)
ax1.set_title('Time series')
ax1.set_xlabel('n')

ax2.plot(w_path, alpha=1)
ax2.set_title('Weights path')
ax2.set_ylabel('w')
ax2.set_xlabel('n')


ax1.plot(p**2)


env = (w_path**2).sum(1)**0.5/2

ax1.plot(env**2, 'b', alpha=0.5)
ax1.plot(i_signal**2, 'g',  alpha=0.9)
ax1.plot(i_envelope**2, 'g',  alpha=0.9)
#ax1.plot(i_envelope, 'g', alpha=0.5)
print((p[:len(i_signal)] - i_signal).std()/((i_signal).std()))
#print((((w_path**2).sum(1)**0.5/2)[:len(i_signal)] - i_envelope).std()/((i_envelope).std()))
ax2.set_xlim(18000, 20000)
ax1.set_xlim(18000, 20000)
plt.tight_layout()
plt.show()
plt.show()