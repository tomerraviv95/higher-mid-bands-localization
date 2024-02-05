from typing import List, Union

import numpy as np

from python_code import conf
from python_code.estimation.algs import ALG_TYPE, ALGS_DICT
from python_code.utils.bands_manipulation import Band
from python_code.utils.basis_functions import compute_time_options
from python_code.utils.constants import BandType, Estimation


class TimeEstimator2D:
    def __init__(self, bands: List[Band]):
        self.multi_band = len(bands) > 1
        if self.multi_band:
            # make sure all the different bands have the same time resolution
            assert all([(band.K / band.BW) == (bands[0].K / bands[0].BW) for band in bands])
        self.times_dict = np.arange(0, bands[0].K / bands[0].BW, conf.T_res)
        if self.multi_band:
            self._multiband_constructor(bands)
        else:
            self._single_band_constructor(bands)

    def _single_band_constructor(self, bands: List[Band]):
        band = bands[0]
        self._time_options = compute_time_options(0, band.K, band.BW, values=self.times_dict)
        if self._time_options.shape[0] < self.times_dict.shape[0]:
            self.times_dict = self.times_dict[:self._time_options.shape[0]]
        self.K = band.K
        self.algorithm = ALGS_DICT[ALG_TYPE][BandType.SINGLE](1.25)

    def _multiband_constructor(self, bands: List[Band]):
        self._time_options = [compute_time_options(0, band.K, band.BW, values=self.times_dict) for band in bands]
        if self._time_options[0].shape[0] < self.times_dict.shape[0]:
            self.times_dict = self.times_dict[:self._time_options[0].shape[0]]
        self.K = [band.K for band in bands]
        self.algorithm = ALGS_DICT[ALG_TYPE][BandType.MULTI](1.25)

    def estimate(self, y: Union[np.ndarray, List[np.ndarray]]) -> Estimation:
        if self.multi_band:
            y = [np.transpose(y_single, [1, 0, 2]) for y_single in y]
        else:
            y = np.transpose(y, [1, 0, 2])
        self._indices, self._spectrum, _ = self.algorithm.run(y=y, n_elements=self.K,
                                                              basis_vectors=self._time_options)
        estimator = Estimation(TOA=self.times_dict[self._indices])
        return estimator
