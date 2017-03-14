import numpy as np

def exp_smooth(x, factor=0.3):
    y = np.zeros_like(x)
    y[0] = x[0]
    for k in range(1, len(y)):
        y[k] = factor * x[k] + (1 - factor) * y[k-1]
    return y

if __name__ == '__main__':
    import pylab as plt
    x = np.sin(np.arange(100)*2*np.pi/50)
    x_n = x + np.random.normal(size=(100, ))*0.2
    plt.plot(x)
    plt.plot(x_n)
    plt.plot(exp_smooth(x_n, factor=0.3))
    plt.show()