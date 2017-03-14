import numpy as np

# main algorithm:
def naive_rls(X, d, mu):
    n_components = X.shape[1]
    w = np.zeros(shape=(n_components, ))
    w_path = np.zeros(shape=(d.shape[0], n_components))
    w_path[0] = w
    d_hat = np.zeros_like(d)
    for k in range(len(d_hat)-1):
        w = w + X[k] * 2 * mu * (d[k] - d_hat[k])
        w_path[k+1] = w
        d_hat[k+1] = np.dot(X[k], w)
    return d_hat, w, w_path