from utils.filters.sin_base_rls import evelope_from_sinbase
from utils.rls.naive import naive_rls
from utils.rls.sparls import sparls
from utils.sinbase import get_base_by_freqs


def naive_rls_envelope_detector(raw, band, fs, freqs, mu):
    n = raw.shape[0]
    base = get_base_by_freqs(n, fs, freqs) / (len(freqs) * 2)
    signal, w, w_path = naive_rls(base, raw, mu=mu)
    envelope = evelope_from_sinbase(base, freqs, fs, w_path)
    return signal, envelope


def sparls_ed(raw, band, fs, freqs, alpha=0.12, lambda_=0.9, K=1, gamma=0., l2=0):
    n = raw.shape[0]
    base = get_base_by_freqs(n, fs, freqs) / (len(freqs) * 2)
    sigma=1
    signal, w, w_path = sparls(base, raw, sigma=1, alpha=alpha, lambda_=lambda_, K=K, gamma=gamma, l2=l2)
    envelope = evelope_from_sinbase(base * (len(freqs) * 2), freqs, fs, w_path)/ (len(freqs) * 2)
    #import pylab as plt
    #plt.plot(signal)
    #plt.plot(envelope)
    #plt.show()
    return signal, envelope