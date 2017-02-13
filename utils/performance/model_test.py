import numpy as np
import pylab as plt

from utils.envelope import RCEnvelopeDetector
from utils.metrics import truncated_nmse


def predictor_performanse(input_series, output_series, predictor, delay, plot=False):
    # collect predictions
    prediction = np.zeros_like(output_series)
    for k in range(len(output_series) - delay):
        prediction[k + delay] = predictor.predict(input_series[k], output_series[k])

    metrics = {
        'nmse': truncated_nmse(prediction, output_series, start_from=0),
        'nmse-second-half': truncated_nmse(prediction, output_series, start_from=None)
    }

    if plot:
        plt.plot(output_series, '.-')
        plt.plot(prediction, '.-')
        plt.fill_between(np.arange(len(output_series)), 0, np.abs(output_series - prediction),
                         facecolor='red', edgecolor='white', alpha=0.5)
        plt.legend(['target', 'prediction', 'abs(error)'])
        plt.show()

    return metrics

if __name__ == '__main__':
    print('*** sin noise test')
    t = np.arange(500) / 500
    x = np.sin(2 * np.pi * t * 10) + np.sin(2 * np.pi * t * np.pi) + 0.01 * np.random.normal(size=500)
    x = x.reshape(-1, 1)
    from utils.models.identity import IdentityPredictor
    from utils.models.rls import DelayedRLSPredictor
    identity =  IdentityPredictor()
    rls = DelayedRLSPredictor(1, delay=10, mu=1)
    for model in [identity, rls]:
        print(predictor_performanse(x, x[:, 0], model, 10))

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
    #data = lfilter(b, a, data, axis=0)

    plt.plot(data + 10 * np.arange(data.shape[1]))
    plt.show()
    delay = 28
    rls = DelayedRLSPredictor(1, delay=delay, mu=1, lambda_=0.9999, M=300)
    for model in [identity, rls]:
        print(predictor_performanse(data[:, 15:16], data[:, 15], model, delay, plot=True, ))
