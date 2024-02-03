from collections import namedtuple

from python_code import conf

Band = namedtuple('Band', ['fc', 'Nr_x', 'Nr_y', 'K', 'BW'])


def get_bands_from_conf():
    """"
    Gather all the hyperparameters per band into a single data holder
    Each band shall hold the frequency fc, number of antennas, number of subcarriers and BW
    """
    bands = []
    for fc, Nr_x, Nr_y, K, BW in zip(conf.fc, conf.Nr_x, conf.Nr_y, conf.K, conf.BW):
        band = Band(fc=fc, Nr_x=Nr_x, Nr_y=Nr_y, K=K, BW=BW)
        bands.append(band)
    return bands
