from collections import namedtuple

from python_code import conf
from python_code.utils.constants import BandType

Band = namedtuple('Band', ['fc', 'Nr', 'K', 'BW'])


def get_bands_from_conf():
    """"
    Gather all the hyperparameters per band into a single data holder
    Each band shall hold the frequency fc, number of antennas, number of subcarriers and BW
    """
    bands = []
    for fc, Nr, K, BW in zip(conf.fc, conf.Nr, conf.K, conf.BW):
        band = Band(fc=fc, Nr=Nr, K=K, BW=BW)
        bands.append(band)
    if len(bands) == 1:
        assert conf.band_type == BandType.SINGLE.name
    else:
        assert conf.band_type == BandType.MULTI.name
    return bands
