import numpy as np
import pylab as plt


# generate data:
np.random.seed(42)
n_steps = 500
n_components = 50
n_act = 5
t = np.arange(n_steps)
X = np.random.normal(size=(n_steps, n_components))
w_opt = np.random.normal(size=(n_components, ))
w_opt[np.random.choice(n_components, n_components - n_act, replace=0)] = 0
d = np.dot(X, w_opt)
#plt.plot(d)
#plt.show()


# LCEM
def lcem(B, u, w, Ip, Im, K, tau):
    n_components = B.shape[0]
    r = np.dot(B[:, Ip], w[Ip]) + np.dot(B[:, Im], w[Im]) + u
    Ip = np.arange(n_components)[r >  tau]
    Im = np.arange(n_components)[r < -tau]
    for l in range(K):
        r = np.dot(B[:, Ip], (r[Ip] - tau)) + np.dot(B[:, Im], (r[Im] - tau)) + u
        Ip = np.arange(n_components)[r > tau]
        Im = np.arange(n_components)[r < -tau]
    for i in range(len(w)):
        if i in Ip:
            w[i] = r[i] - tau
        elif i in Im:
            w[i] = r[i] + tau
        else:
            w[i] = 0
    return w, Ip, Im

# main algorithm:
def sparls(X, d, sigma=1, alpha=0.01, lambda_=0.999, K=50, gamma=30):
    n_components = X.shape[1]
    kappa = (alpha/sigma)**2
    I = np.eye(n_components)
    B = I - kappa * np.dot(X[:1].T, X[:1])
    u = kappa * X[0] * d[0]
    w = np.zeros(shape=(n_components, ))
    w_path = np.zeros(shape=(d.shape[0], n_components))
    w_path[0] = w
    d_hat = np.zeros_like(d)
    Ip = np.arange(n_components)
    Im = np.arange(n_components)
    for n, x in enumerate(X[1:]):
        B = lambda_ * B - kappa * np.dot(x[:, None], x[None, :]) + (1 - lambda_)*I
        u = lambda_ * u + kappa * d[n+1] * x
        w, Ip, Im = lcem(B, u, w, Ip, Im, K, tau=gamma*alpha**2)
        d_hat[n+1] = np.dot(x, w)
        w_path[n+1] = w
    return d_hat, w, w_path

p, w, w_path = sparls(X, d)


f, [ax1, ax2] = plt.subplots(2)
ax1.plot(p)
ax1.plot(d)
ax1.set_title('Time series')
ax1.set_xlabel('n')

ax2.plot(w_path, 'b', alpha=1)
ax2.plot(w_path*0 + w_opt, 'g', alpha=1)
ax2.set_title('Weights path')
ax2.set_ylabel('w')
ax2.set_xlabel('n')


plt.tight_layout()
plt.show()
