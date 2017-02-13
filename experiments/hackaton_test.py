import h5py
import pylab as plt
import numpy as np
with h5py.File('fb_data.h5') as f:
    for dataset in f:
        data = f[dataset][:]
        plt.plot(data[0] + np.arange(21)*5, 'g')
        plt.title(dataset)
        plt.show()