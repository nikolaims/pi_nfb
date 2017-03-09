import numpy as np

# LCEM
def lcem(B, u, w, Ip, Im, K, tau, l2):
    n_components = B.shape[0]
    r = np.dot(B[:, Ip], w[Ip]) + np.dot(B[:, Im], w[Im]) + u
    Ip = np.arange(n_components)[r >  tau]
    Im = np.arange(n_components)[r < -tau]
    for l in range(K):
        r = np.dot(B[:, Ip], (r[Ip] - tau)) + np.dot(B[:, Im], (r[Im] - tau)) + u
        Ip = np.arange(n_components)[r > tau]
        Im = np.arange(n_components)[r < -tau]
    for i in range(len(w)):
        ll = l2*w[i]
        if i in Ip:
            w[i] = r[i] - tau
        elif i in Im:
            w[i] = r[i] + tau
        else:
            w[i] = 0
        w[i]-=ll

    return w, Ip, Im

# main algorithm:
def sparls(X, d, sigma=1., alpha=0.02, lambda_=0.95, K=1, gamma=100, l2=0.):
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
        w, Ip, Im = lcem(B, u, w, Ip, Im, K, tau=gamma*alpha**2, l2=l2)
        d_hat[n+1] = np.dot(x, w)
        w_path[n+1] = w
    return d_hat, w, w_path