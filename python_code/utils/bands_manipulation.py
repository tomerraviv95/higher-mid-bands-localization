from collections import namedtuple

from python_code import conf

Band = namedtuple('Band', ['fc', 'Nr_x', 'Nr_y', 'K', 'BW'])


def get_bands_from_conf():
    bands = []
    for fc, Nr_x, Nr_y, K, BW in zip(conf.fc, conf.Nr_x, conf.Nr_y, conf.K, conf.BW):
        band = Band(fc=fc, Nr_x=Nr_x, Nr_y=Nr_y, K=K, BW=BW)
        bands.append(band)
    return bands
