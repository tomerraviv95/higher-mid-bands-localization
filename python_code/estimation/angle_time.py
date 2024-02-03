import numpy as np

from python_code.estimation import Estimation
from python_code.estimation.algs import ALG_TYPE, ALGS_DICT
from python_code.estimation.angle import AngleEstimator2D, AngleEstimator3D
from python_code.estimation.time import TimeEstimator2D, TimeEstimator3D
from python_code.utils.bands_manipulation import Band


class AngleTimeEstimator2D:
    def __init__(self, band: Band):
        self.angle_estimator = AngleEstimator2D(band)
        self.time_estimator = TimeEstimator2D(band)
        self.angle_time_options = np.kron(self.angle_estimator._angle_options, self.time_estimator._time_options)
        self.algorithm = ALGS_DICT[ALG_TYPE](1.5)
        self.Nr_x = band.Nr_x
        self.K = band.K

    def estimate(self, y: np.ndarray) -> Estimation:
        self.indices, self._spectrum, _ = self.algorithm.run(y=y, n_elements=self.Nr_x * self.K,
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


class AngleTimeEstimator3D:
    def __init__(self, band: Band):
        self.angle_estimator = AngleEstimator3D(band)
        self.time_estimator = TimeEstimator3D(band)
        print(111)
        self.angle_time_options = np.kron(self.angle_estimator._angle_options.astype(np.complex64),
                                          self.time_estimator._time_options.astype(np.complex64))
        print(222)
        self.algorithm = ALGS_DICT[ALG_TYPE](10)

    def estimate(self, y: np.ndarray) -> Estimation:
        n_elements = self.angle_estimator.Nr_x * self.angle_estimator.Nr_y * self.time_estimator.K
        self.indices, self._spectrum, _ = self.algorithm.run(y=y, n_elements=n_elements,
                                                             basis_vectors=self.angle_time_options,
                                                             second_dim=len(self.angle_estimator.zoa_angles_dict),
                                                             third_dim=len(self.time_estimator.times_dict), batches=10)
        # if no peaks found - return an empty estimation
        if len(self.indices) == 0:
            return Estimation()
        self._aoa_indices = self.indices[:, 0]
        self._zoa_indices = self.indices[:, 1]
        self._toa_indices = self.indices[:, 2]
        estimator = Estimation(AOA=self.angle_estimator.aoa_angles_dict[self._aoa_indices],
                               ZOA=self.angle_estimator.zoa_angles_dict[self._zoa_indices],
                               TOA=self.time_estimator.times_dict[self._toa_indices])
        return estimator
