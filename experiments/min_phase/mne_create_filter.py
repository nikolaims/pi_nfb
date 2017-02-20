from mne1.filter import create_filter
from mne1.viz import plot_filter
from utils.data.loaders import get_signal

x = get_signal()[:, 4]
taps = create_filter(x, 250, 8, None, filter_length=100, phase='minimum')
plot_filter(taps, 250)