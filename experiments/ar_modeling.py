from utils.data.loaders import get_ideal_signal
from utils.metrics import truncated_nmse
import pylab as plt
import numpy as np
from scipy.signal import *
from utils.filters import dc_blocker

causal=True
data = get_ideal_signal(causal=causal)


print('statmodels')
#plt.plot(data[:10000+1000, 15])
import statsmodels.api as sm
ar_model = sm.tsa.AR(data[:10000, 15]).fit(10, transparams=True)
print(ar_model.params)
#plt.plot(range(10000, 10000 + 1000), ar_model.predict(10000, 10000 + 1000 - 1))
#plt.ylim(-10, 10)

print('linear reg')
train = data[:20000, 15]
M = 100
lag = 28
x_train = np.array([train[k:k+M] for k in range(len(train)-M - lag)])
y_train = train[M+lag:]

from sklearn.linear_model import LinearRegression, ElasticNetCV, RidgeCV
reg = ElasticNetCV()
reg.fit(x_train, y_train)
print(reg.coef_)

x0 = train[:M+lag]
plt.plot(train)
plt.vlines(M+lag - 1, train.min(), train.max())
plt.plot(range(M+lag, len(train)), reg.predict(x_train))
r_prediction = list(x0)
for k in range(M+lag, len(train)):
    if r_prediction[-1] < 100:
        r_prediction.append(reg.predict(np.array(r_prediction[-M-lag:-lag]).reshape(1, -1))[0])
    else:
        r_prediction.append(100)

plt.plot(r_prediction,alpha=0.8)
plt.ylim([train.min(), train.max()])
plt.xlim([0, 750])
plt.legend(['target {}'.format('filtfilt' if not causal else 'causal FIR'),
            '{} steps ahead prediction'.format(lag), 'recursive {} steps ahead prediction'.format(lag)])
plt.show()