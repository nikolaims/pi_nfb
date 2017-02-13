import numpy as np


def truncated_nmse(prediction, target, start_from=None):
    if start_from is None:
        start_from = len(target)//2
    errors = prediction[start_from:] - target[start_from:]
    return np.mean(errors**2)/np.var(target[start_from:])