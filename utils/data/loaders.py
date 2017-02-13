import h5py
import numpy as np
from os.path import dirname, realpath

from scipy.signal import butter, filtfilt, lfilter

from utils.filters import dc_blocker, magic_filter_taps

data_dir = dirname(dirname(dirname(realpath(__file__)))) + '/data/'

def load_feedback(ica_artifact=False, csp_alpha=False, signal_name='left'):
    with h5py.File(data_dir + 'experiment_data.h5', 'r') as f: #TODO: path
        protocol = 'protocol10'
        raw = f[protocol+'/raw_data'][:]
        print('Data was loaded from {} "{}"'.format(protocol, f[protocol].attrs['name']))
        signals_names = list(f[protocol+'/signals_stats'].keys())
        derived = f[protocol+'/signals_data'][:][:, signals_names.index(signal_name)]
        _rejections_group = f[protocol+'/signals_stats/{}/rejections'.format(signal_name)]
        rejections = [_rejections_group['rejection{}'.format(k + 1)][:] for k in range(len(_rejections_group)//2)]
        left_spatial_filter = f[protocol+'/signals_stats/{}/spatial_filter'.format(signal_name)][:]

    data = raw
    if ica_artifact:
        data = np.dot(data, rejections[0])
    if csp_alpha:
        data = np.dot(data, rejections[1])
    signal = np.dot(data, left_spatial_filter)
    return data, signal, derived

def get_ideal_signal(band = (8, 12), causal=False, causal_iir=True, b_order=4, min_phase=False):
    data, signal, derived = load_feedback(ica_artifact=True)
    data = dc_blocker(data)
    nq = 125
    if min_phase:
        from utils.filters import min_phase_magic_filter
        return lfilter(min_phase_magic_filter(), 1.0, data, axis=0)

    b, a = butter(b_order, [band[0] / nq, band[1] / nq], 'band')
    if causal:
        if not causal_iir:
            data = lfilter(magic_filter_taps(), 1.0, data, axis=0)
        else:
            data = lfilter(b, a, data, axis=0)
    else:

        data = filtfilt(b, a, data, 0)
    return data

def get_signal():
    data, signal, derived = load_feedback(ica_artifact=True)
    data = dc_blocker(data)
    return data