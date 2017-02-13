from collections import deque
import numpy as np

class DelayedRLSPredictor:
    def __init__(self, n_channels, M=3, lambda_=0.999, delta=100, delay=0, mu=0.3): #, target_channel=15):
        self._M = M
        self._lambda = lambda_
        self._delay = delay
        self._mu = mu
        size = M * n_channels
        self._w = np.zeros((size,))
        self._P = delta * np.eye(size)
        self.regressors = deque(maxlen=M + delay)
        #self.target_channel = target_channel

    def predict(self, current_input_sample, current_output_sample):
        self.regressors.append(current_input_sample)
        regressors = np.array(self.regressors)
        if regressors.shape[0] >= self._delay + self._M:
            # predicted var x(t)
            predicted = current_output_sample # regressors[-1, self.target_channel]

            # predictor var [x(t - M - delay), x(t - M + 1 - delay), ..., x(t - delay)]
            predictor = regressors[- self._M - self._delay: - self._delay if self._delay>0 else None].flatten()  #

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
    rls = DelayedRLSPredictor(n_channels=1, delay=delay, M=20)

    x_pred = np.zeros_like(x)
    for k in range(len(t) - delay):
        x_pred[k + delay] = rls.predict(x[k], x[k][0])

    plt.plot(t, x, '.-')
    plt.plot(t, x_pred, '.-')
    plt.show()