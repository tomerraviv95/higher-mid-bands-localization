from typing import List, Union

import numpy as np

from python_code import conf
from python_code.estimation.algs import ALGS_DICT, ALG_TYPE
from python_code.utils.bands_manipulation import Band
from python_code.utils.basis_functions import compute_angle_options
from python_code.utils.constants import BandType
from python_code.utils.constants import Estimation


class AngleEstimator:
    """
    Angles Estimator for the AOA
    """

    def __init__(self, bands: List[Band]):
        self.aoa_angles_dict = np.arange(-np.pi / 2, np.pi / 2, conf.aoa_res * np.pi / 180)
        self.multi_band = len(bands) > 1
        if self.multi_band:
            self._multiband_constructor(bands)
        else:
            self._single_band_constructor(bands)

    def _single_band_constructor(self, bands: List[Band]):
        band = bands[0]
        self._angle_options = compute_angle_options(np.sin(self.aoa_angles_dict), values=np.arange(band.Nr))
        self.algorithm = ALGS_DICT[ALG_TYPE][BandType.SINGLE]()

    def _multiband_constructor(self, bands: List[Band]):
        self._angle_options = [compute_angle_options(np.sin(self.aoa_angles_dict), values=np.arange(band.Nr)) for band
                               in bands]
        self.algorithm = ALGS_DICT[ALG_TYPE][BandType.MULTI]()

    def estimate(self, y: Union[np.ndarray, List[np.ndarray]]) -> Estimation:
        self._indices, self._spectrum = self.algorithm.run(y=y, basis_vectors=self._angle_options)
        estimator = Estimation(AOA=self.aoa_angles_dict[self._indices])
        return estimator
