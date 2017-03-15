from utils.filters.sin_base_rls import evelope_from_sinbase
from utils.rls.naive import naive_rls
from utils.sinbase import get_base_by_freqs


def naive_rls_envelope_detector(raw, band, fs, freqs, mu):
    n = raw.shape[0]
    base = get_base_by_freqs(n, fs, freqs) / (len(freqs) * 2)
    signal, w, w_path = naive_rls(base, raw, mu=mu)
    envelope = evelope_from_sinbase(base, freqs, fs, w_path)
    return signal, envelope