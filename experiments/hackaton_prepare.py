import xml.etree.ElementTree as ET

def get_lsl_info_from_xml(xml_str_or_file):
    try:
        tree = ET.parse(xml_str_or_file)
        root = tree.getroot()
    except FileNotFoundError:
        root = ET.fromstring(xml_str_or_file)
    info = {}
    channels = [k.find('label').text for k in root.find('desc').find('channels').findall('channel')]
    fs = int(root.find('nominal_srate').text)
    return channels, fs

import numpy as np
import pylab as plt

from scipy.signal import *

import pandas as pd
import seaborn as sns
import pickle
from utils.filters import magic_filter_taps


pilot_dir = 'C:\\Users\\Nikolai\\Downloads\\pilot'




experiments1 = ['pilot_Nikolay_1_10-17_13-57-56', #BAD NO FILTERS
                'pilot_Plackhin_1_10-20_12-03-01',
                'pilot_Tatiana_1_10-17_15-04-39',
                'pilot_Polyakova_1_10-24_15-21-18',
                'pilot_Emelyannikov29_1_11-09_20-14-28']

experiments2 = ['pilot_Nikolay_2_10-18_14-57-23',
                'pilot_Plackhin_2_10-21_13-09-27',
                'pilot_Tatiana_2_10-18_16-00-44',
                'pilot_Polyakova_2_10-25_14-19-56',
                'pilot_Emelyannikov29_2_11-10_19-47-25']

experiments = experiments1 + experiments2


PROTOCOLS = {'FB': [3, 5, 7, 9, 11],
             'BASELINE': 2,
             'ROTATE': [1, 13],
             'ALL': list(range(1, 15)),
             'NAMES': ['Filters', 'Rotate', 'Baseline', 'FB', 'Rest', 'FB', 'Rest', 'FB', 'Rest', 'FB', 'Rest', 'FB',
                       'Rest', 'Rotate'],
             'N_SAMPLES': [30000, 15000, 7500, 30000, 5000, 30000, 5000, 30000, 5000, 30000, 5000, 30000, 5000, 15000]}

fs = 250

import h5py
results = {}

channel = 'C3'
n_samples = 7500


use_pz = False
reject_alpha = True
for experiment in experiments[:]:
    print('\n\nEXPERIMENT', experiment)


    with h5py.File('{}\\{}\\{}'.format(pilot_dir, experiment, 'experiment_data.h5')) as f:
        rejections = [f['protocol1/signals_stats/left/rejections/rejection{}'.format(j + 1)][:]
                      for j in range(2)]

    rejection= rejections[0]
    if reject_alpha:
        rejection = np.dot(rejection, rejections[1])


    # load data
    with h5py.File('{}\\{}\\{}'.format(pilot_dir, experiment, 'experiment_data.h5')) as f:
        labels, fs = get_lsl_info_from_xml(f['stream_info.xml'][0])
        print('fs: {}\nall labels {}: {}'.format(fs, len(labels), labels))
        channels = [label for label in labels if label not in ['A1', 'A2', 'AUX']]
        pz_index = channels.index('Pz')
        print('selected channels {}: {}'.format(len(channels) - use_pz, channels))
        data = []
        for j in PROTOCOLS['ALL']:
            raw = f['protocol{}/raw_data'.format(j)][:]
            raw = raw[:raw.shape[0] - raw.shape[0] % fs]
            if use_pz:
                raw = raw[:, np.arange(raw.shape[1]) != pz_index]
            data.append(np.dot(raw, rejection))
        assert [f['protocol{}'.format(j)].attrs['name'] for j in PROTOCOLS['ALL']] == PROTOCOLS['NAMES'], 'bad pr names'
        if [dt.shape[0] for dt in data] != PROTOCOLS['N_SAMPLES']:
            print('WARN: bad samples number for {}:\nexpected\n{},\nget\n{}'.format(experiment, PROTOCOLS['N_SAMPLES'],
                                                                    [dt.shape[0] for dt in data]))

        results[experiment] = data

taps = magic_filter_taps()
from scipy.signal import lfilter
for experiment in experiments:
    fb_data = np.zeros((5, 30000 - 56, 21))
    for i, prt in enumerate(PROTOCOLS['FB']):
        data = lfilter(taps, 1.0, results[experiment][prt], axis=0)[56:]
        print(data.mean(0).shape)
        fb_data[i] = (data - data.mean(0)) / data.std(0)
    print(fb_data.shape)
    with h5py.File('fb_data.h5') as f:
        f.create_dataset(experiment, data=fb_data)
