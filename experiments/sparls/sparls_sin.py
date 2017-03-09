import numpy as np
import pylab as plt


# generate data:
from utils.rls.sparls import sparls

np.random.seed(42)
n_steps = 5000
n_components = 8
n_act = 3
t = np.arange(n_steps)
fs = 500
freqs = np.linspace(9, 11, n_components//2)
X = []
for fq in freqs:
    X += [np.sin(2*fq * np.pi*t/fs), np.cos(2*fq * np.pi*t/fs)]
X = np.array(X).T/n_components
w_opt = np.zeros(shape=(n_steps, n_components)) + np.random.normal(size=(n_components, ))

first_half_choice = np.random.choice(n_components, n_components - n_act, replace=0)
second_half_choice = np.random.choice(n_components, n_components - n_act, replace=0)
w_opt[:n_steps//2, first_half_choice] = 0
w_opt[n_steps//2:, second_half_choice] = 0
sigma = 0.01

d_clear = np.sum(X * w_opt, 1)
d = d_clear + np.random.normal(size=n_steps, scale=sigma)
print('SNR', sigma / d_clear.std())
#plt.plot(d)
#plt.show()

p, w, w_path = sparls(X, d, sigma=sigma, alpha=sigma/2, lambda_=0.99, K=1, gamma=1/sigma)


f, [ax1, ax2] = plt.subplots(2)
#ax1.semilogy((p-d_clear)**2)
#ax1.plot(d)
ax1.set_title('Time series')
ax1.set_xlabel('n')

ax2.plot(w_path, alpha=1)
ax2.plot(w_opt, 'k', alpha=1)
ax2.set_title('Weights path')
ax2.set_ylabel('w')
ax2.set_xlabel('n')

ax1.plot(d_clear)
ax1.plot(p)
ax1.plot(d, alpha=0.5)

plt.tight_layout()
plt.show()
