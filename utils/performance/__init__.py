import numpy as np
import pylab as plt

def sin_noise_test(predictor, n=500, fs=500, delay=10, plot=False):
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * t * 10) + np.sin(2 * np.pi * t * np.pi) + 0.01 * np.random.normal(size=500)
    x = x.reshape(-1, 1)
    x_pred = np.zeros_like(x)
    for k in range(len(t) - delay):
        x_pred[k + delay] = predictor.predict(x[k], x[k][0])

    print('RMSE:\t', np.mean((x - x_pred)**2)/np.var(x))
    print('RMSE-half:\t', np.mean((x - x_pred)[n//2:] ** 2) / np.var(x))

    if plot:
        plt.plot(t, x, '.-')
        plt.plot(t, x_pred, '.-')
        plt.legend(['target', 'prediction'])
        plt.show()


if __name__ == '__main__':
    from utils.models.rls import DelayedRLSPredictor
    delay = 10
    rls = DelayedRLSPredictor(n_channels=1, delay=delay, M=20)
    sin_noise_test(rls, plot=True)