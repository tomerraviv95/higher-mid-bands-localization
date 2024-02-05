from typing import List

import numpy as np

from python_code.estimation.algs import ALG_TYPE, ALGS_DICT
from python_code.utils.bands_manipulation import Band
from python_code.utils.constants import BandType, Estimation
from python_code.estimation.parameters_2d.angle import AngleEstimator2D
from python_code.estimation.parameters_2d.time import TimeEstimator2D

coef_per_frequencies_dict = {6000: 2.5, 24000: 1.5}


class AngleTimeEstimator2D:
    def __init__(self, bands: List[Band]):
        self.angle_estimator = AngleEstimator2D(bands)
        self.time_estimator = TimeEstimator2D(bands)
        if self.angle_estimator.multi_band:
            self.angle_time_options = [
                np.kron(self.angle_estimator._angle_options[i], self.time_estimator._time_options[i]) for i in
                range(len(bands))]
            self.algorithm = ALGS_DICT[ALG_TYPE][BandType.MULTI](coef_per_frequencies_dict[bands[0].fc])
            self.n_elements = [band.Nr_x * band.K for band in bands]
        else:
            self.angle_time_options = np.kron(self.angle_estimator._angle_options, self.time_estimator._time_options)
            self.algorithm = ALGS_DICT[ALG_TYPE][BandType.SINGLE](coef_per_frequencies_dict[bands[0].fc])
            self.n_elements = bands[0].Nr_x * bands[0].K

    def estimate(self, y: np.ndarray) -> Estimation:
        self.indices, self._spectrum, _ = self.algorithm.run(y=y, n_elements=self.n_elements,
                                                             basis_vectors=self.angle_time_options,
                                                             second_dim=len(self.time_estimator.times_dict))
        # if no peaks found - return an empty estimation
        if len(self.indices) == 0:
            return Estimation()
        self._aoa_indices = self.indices[:, 0]
        self._toa_indices = self.indices[:, 1]
        estimator = Estimation(AOA=self.angle_estimator.aoa_angles_dict[self._aoa_indices],
                               TOA=self.time_estimator.times_dict[self._toa_indices])
        return estimator
