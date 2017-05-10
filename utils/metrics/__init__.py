import numpy as np
import pylab as plt

def nmse(prediction, target, start_from=None):
    if start_from is None:
        start_from = 0
    errors = prediction[start_from:] - target[start_from:]
    return np.mean(errors**2)/np.var(target[start_from:])

def smoothness(x, ideal=None):
    if ideal is None:
        return np.diff(x).std() / np.abs(np.diff(x).mean())
    else:
        return np.diff(x).std() / np.diff(ideal).std()

def find_lag(x, target, fs=None, show=False, nmse=False):
    n = 10000
    nor = lambda x:  (x - np.mean(x)) / np.std(x)
    lags = np.arange(n)
    mses = np.zeros_like(lags).astype(float)
    n_points = len(target) - n

    for lag in lags:
        if nmse:
            mses[lag] = np.mean((nor(target[:n_points]) - nor(x[lag:n_points+lag]))**2)
        else:
            mses[lag] = -np.mean(nor(target[:n_points])*nor(x[lag:n_points+lag]))
    lag = np.argmin(mses)



    if show:
        f, (ax1, ax2) = plt.subplots(2)
        ax1.plot(mses)
        ax1.plot(lag, np.min(mses), 'or')
        lag_str = '{}'.format(lag) if fs is None else '{} ({:.3f} s)'.format(lag, lag/fs)
        ax1.text(lag+n//100*2, np.min(mses), lag_str)
        ax2.plot(nor(target))
        ax2.plot(nor(x[lag:]), alpha=1)
        ax2.plot(nor(x), alpha=0.5)
        ax2.legend(['target',  'x[{}:]'.format(lag), 'x'])
        plt.show()
    return lag

def lag_compensed_nmse(x, ideal, show=False):
    lag = find_lag(x, ideal, show=show)
    return lag, nmse(x[lag:], ideal[:-lag])

if __name__ == '__main__':
    n = 2000
    lag = 2
    x = np.sin(np.arange(n)*2*np.pi/250) + np.arange(n)/n
    x_noise = x[:-lag] + np.random.normal(size=(n-lag, ))*0.01
    x = x[lag:]
    plt.plot(x)
    plt.plot(x_noise)
    plt.show()
    print('smoothness', smoothness(x), smoothness(x_noise), smoothness(x_noise, x))
    print('nmse', lag_compensed_nmse(x_noise, x, 1), nmse(x_noise[lag:], x[:-lag]))