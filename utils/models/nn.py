from collections import deque

import lasagne
import theano.tensor as T
import theano
import numpy as np

class NNPredictor:
    def __init__(self, delay=2, n_channels=32, M=3):
        input_var = T.tensor3('inputs')
        target_var = T.vector('targets')
        l_in = lasagne.layers.InputLayer(shape=(None, M, n_channels), input_var=input_var)
        l_out = lasagne.layers.DenseLayer(l_in, num_units=1, nonlinearity=lasagne.nonlinearities.linear)
        p = lasagne.layers.get_output(l_out)
        self.M = M
        loss = lasagne.objectives.squared_error(p, target_var)
        loss = loss.mean()
        params = lasagne.layers.get_all_params(l_out, trainable=True)
        loss = loss #+ lasagne.regularization.regularize_layer_params(l_out, lasagne.regularization.l2)
        updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.001, momentum=0.5)
        self.train_fn = theano.function([input_var, target_var], p, updates=updates)
        self.get_prediction = theano.function([input_var], p)
        self.get_params = theano.function([], params)
        deque_len = M + delay
        self.regressors = deque([np.zeros((n_channels, ))] * deque_len,  maxlen=deque_len)

    def predict(self, current_input_sample, current_output_sample):
        self.regressors.append(current_input_sample)

        regressors = np.array(self.regressors)
        p = self.train_fn(regressors[:self.M][None, :, :], np.array(current_output_sample).reshape((1, )))
        # print(np.array(self.get_params()))
        p = self.get_prediction(regressors[-self.M:][None, :, :])
        return p

if __name__ == '__main__':
    from utils.performance.model_test import predictor_performanse
    print('*** sin noise test')
    t = np.arange(2000) / 500
    x = np.sin(2 * np.pi * t * 10) + np.sin(2 * np.pi * t * np.pi) + 0.01 * np.random.normal(size=2000)
    x = x.reshape(-1, 1)


    nn = NNPredictor(n_channels=1, M=100, delay=10)

    predictor_performanse(x, x[:, 0], nn, 10, plot=True)

    print('*** real data test')
    from utils.data.loaders import load_feedback
    from utils.filters import dc_blocker

    data, _signal, _derived = load_feedback(ica_artifact=True)
    data = dc_blocker(data)
    fs = 250
    nq = 250 / 2

    from scipy.signal import butter, lfilter, filtfilt, hilbert, firwin
    from utils.filters import magic_filter_taps, custom_fir

    b, a = butter(2, [8 / nq, 12 / nq], btype='band')

    # target_series = filtfilt(b, a, data[:, 15], axis=0)
    data = lfilter(magic_filter_taps(), 1.0, data, axis=0)

    data = (data - data.mean(0)) / data.std(0)
    # data = lfilter(b, a, data, axis=0)
    import pylab as plt
    plt.plot(data + 10 * np.arange(data.shape[1]))
    plt.show()
    delay = 28
    nn = NNPredictor(n_channels=1, M=100, delay=delay)
    predictor_performanse(data[:5000, 15:16], data[:5000, 15], nn, delay, plot=True)
