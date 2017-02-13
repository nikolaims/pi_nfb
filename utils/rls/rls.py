from collections import deque

import numpy as np


def rls_predict(u, d, M=10, lambda_=0.9999, delta_var=1000):
    # helpers
    N = len(u)
    delta = delta_var * np.var(u)

    # initializing
    w = np.zeros((M,))
    P = delta * np.eye(M)

    # prediction
    p = np.zeros_like(d)

    # iterations
    for n in range(M, N):
        pi = np.dot(u[n - M:n], P)
        k = pi / (lambda_ + np.dot(pi, u[n - M:n]))
        p[n] = np.dot(w, u[n - M:n])
        w = w + (d[n] - p[n]) * k
        P = 1 / lambda_ * (P - np.dot(k[:, None], pi[None, :]))
    return p


def rls_predict_many(u, d, M=10, lambda_=0.9999, delta_var=1000, show_stats=False, add_one=False, mu=0.25):
    # helpers
    N = len(u)
    delta = delta_var * np.var(u)

    # initializing
    size = (M + int(add_one)) * u.shape[1]
    w = np.zeros((size,))
    P = delta * np.eye(size)

    #
    # add1 = (lambda x: np.hstack([np.array([1]), x]).flatten()) if add_one else (lambda x: x.flatten())
    add1 = lambda x: x.flatten()

    # prediction
    p = np.zeros_like(d)

    # iterations
    for n in range(M, N):
        uu = add1(u[n - M:n])
        pi = np.dot(uu, P)
        k = pi / (lambda_ + np.dot(pi, uu))
        p[n] = np.dot(w, uu)
        w = w + 2 * mu * (d[n] - p[n]) * k
        P = 1 / lambda_ * (P - np.dot(k[:, None], pi[None, :]))
        if show_stats:
            print(sum(w))
    return p


class DelayedRLSPredictor:
    def __init__(self, n_channels, M=3, lambda_=0.999, delta=100, delay=0, mu=0.3, target_channel=15):
        self._M = M
        self._lambda = lambda_
        self._delay = delay
        self._mu = mu
        size = M * n_channels
        self._w = np.zeros((size,))
        self._P = delta * np.eye(size)
        self.regressors = deque(maxlen=M + delay + 1)
        self.target_channel = target_channel

    def predict(self, sample):
        self.regressors.append(sample)
        regressors = np.array(self.regressors)
        if regressors.shape[0] > self._delay + self._M:
            # predicted var x(t)
            predicted = regressors[-1, self.target_channel]

            # predictor var [x(t - M - delay), x(t - M + 1 - delay), ..., x(t - delay)]
            predictor = regressors[- self._M - self._delay - 1: - self._delay - 1].flatten()  #

            # update helpers
            pi = np.dot(predictor, self._P)
            k = pi / (self._lambda + np.dot(pi, predictor))
            self._P = 1 / self._lambda * (self._P - np.dot(k[:, None], pi[None, :]))

            # update weights
            dw = (predicted - np.dot(self._w, predictor)) * k
            self._w = self._w + self._mu * dw

            # return prediction x(t + delay)
            return np.dot(self._w, regressors[- self._M:].flatten())

        # if lenght of regressor less than M + delay + 1 return 0
        return 0


if __name__ == '__main__':
    import pylab as plt
    t = np.arange(500)/500
    x = np.sin(2*np.pi * t *10) + np.sin(2*np.pi * t *np.pi) + 0.01*np.random.normal(size=500)
    x = x.reshape(-1, 1)
    print(x.shape)
    delay = 10
    rls = DelayedRLSPredictor(n_channels=1, delay=delay, target_channel=0, M=20)

    x_pred = np.zeros_like(x)
    for k in range(len(t) - delay):
        x_pred[k + delay] = rls.predict(x[k])

    plt.plot(t, x, '.-')
    plt.plot(t, x_pred, '.-')
    plt.show()