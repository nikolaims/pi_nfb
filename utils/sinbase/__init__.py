import numpy as np
from scipy.signal import lfilter
from sklearn.linear_model import LassoCV, ElasticNetCV, RidgeCV

import pylab as plt
#import seaborn as sns
import pandas as pd

from utils.filters.ideal_filter import get_fir_filter


def elastic_net_base(x, fs, n_components, band):
    base, freqs = get_base(len(x), fs, n_components, low=band[0], high=band[1], nonlinear=0)
    model = ElasticNetCV(l1_ratio=[0, 0.25, 0.5, 0.75,  1])
    model.fit(base, x)
    print(model.l1_ratio_)
    f, (ax1, ax2) = plt.subplots(2)
    ax1.plot(x)
    ax1.plot(model.predict(base))
    c = np.abs(model.coef_) > 0.00000000001
    # c = model.coef_
    df = pd.DataFrame({'type': ['sin', 'cos']*n_components, 'freq': [freqs[k//2] for k in range(2*n_components)], 'val': c, }).pivot("type", 'freq', 'val')
    sns.heatmap(df, cbar=0, ax=ax2, square=1, xticklabels=['' if k%((n_components-1)//5) else '{:.2f}'.format(freqs[k]) for k in range(n_components)])
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()
    print(sum(model.coef_>0))



def get_base(size, fs, n_components=10, low=7, high=14, nonlinear=1.1, show_freqs=False):
    t = np.arange(size) / fs
    base = []
    if nonlinear>0:
        n = np.linspace(0, 1, n_components//2+1) ** nonlinear
        n = np.hstack((-n[::-1], n[1:]))[1:]
        freqs = n * (high - low) / 2 + (high +low) / 2
    else:
        freqs = np.linspace(low, high, n_components, )

    if show_freqs:
        import pylab as plt
        plt.plot(freqs, np.ones_like(freqs), 'o')
        plt.show()
        print(freqs)
    for model_freq in freqs:
        base += [np.sin(model_freq * t * 2 * np.pi), np.cos(model_freq * t * 2 * np.pi)]
    base = np.vstack(base).T
    return base, freqs


if __name__ == '__main__':
    from utils.data.loaders import get_signal
    from utils.filters.main_freq import get_main_freq
    x = get_signal()[:, 4]
    fs = 250
    main_freq = get_main_freq(x, fs, band=(8, 12))
    i_taps, i_delay = get_fir_filter(fs, main_freq, show=0)
    x = lfilter(i_taps, [1.], x)[i_delay:]
    n_samples = 1000
    for k in range(n_samples, len(x), n_samples):
        elastic_net_base(x[k-n_samples:k], 250, n_components=150, band=(main_freq-1, main_freq+1))

