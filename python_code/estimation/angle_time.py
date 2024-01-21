import numpy as np

from python_code import conf
from python_code.estimation import Estimation
from python_code.estimation.algs import ALG_TYPE, ALGS_DICT
from python_code.estimation.angle import AngleEstimator2D, AngleEstimator3D
from python_code.estimation.time import TimeEstimator2D, TimeEstimator3D
from python_code.utils.peaks_filtering import merge_two, filter_peaks, merge_three

PROXIMITY_THRESH = 15


class AngleTimeEstimator2D:
    def __init__(self):
        self.angle_estimator = AngleEstimator2D()
        self.time_estimator = TimeEstimator2D()
        self.angle_time_options = np.kron(self.angle_estimator._angle_options, self.time_estimator._time_options)
        self.algorithm = ALGS_DICT[ALG_TYPE]

    def estimate(self, y):
        indices, self._spectrum, L_hat = self.algorithm.run(y=y, n_elements=conf.Nr_x * conf.K,
                                                            basis_vectors=self.angle_time_options)
        aoa_indices = indices // conf.T_res
        toa_indices = indices % conf.T_res
        merged = np.array(merge_two(aoa_indices, toa_indices))
        peaks = filter_peaks(merged, L_hat)
        self._aoa_indices = peaks[:, 0]
        self._toa_indices = peaks[:, 1]
        estimator = Estimation(AOA=self.angle_estimator.angles_dict[self._aoa_indices],
                               TOA=self.time_estimator.times_dict[self._toa_indices])
        return estimator


class AngleTimeEstimator3D:
    def __init__(self):
        self.angle_estimator = AngleEstimator3D()
        self.time_estimator = TimeEstimator3D()
        assert len(self.angle_estimator._angle_options) * len(self.time_estimator._time_options) < 10 ** 6
        self.angle_time_options = np.kron(self.angle_estimator._angle_options, self.time_estimator._time_options)
        self.algorithm = ALGS_DICT[ALG_TYPE]

    def estimate(self, y):
        indices, self._spectrum, L_hat = self.algorithm.run(y=y, n_elements=conf.Nr_x * conf.Nr_y * conf.K,
                                                            basis_vectors=self.angle_time_options, do_one_calc=False)
        if len(indices) == 0:
            raise ValueError("No sources detected. Change hyperparameters.")
        angle_indices = indices // conf.T_res
        aoa_indices = angle_indices // (conf.zoa_res)
        zoa_indices = angle_indices % (conf.zoa_res)
        toa_indices = indices % conf.T_res
        merged = np.array(merge_three(aoa_indices, zoa_indices, toa_indices))
        peaks = filter_peaks(merged, L_hat)
        self._aoa_indices = peaks[:, 0]
        self._zoa_indices = peaks[:, 1]
        self._toa_indices = peaks[:, 2]
        estimator = Estimation(AOA=self.angle_estimator.aoa_angles_dict[self._aoa_indices],
                               ZOA=self.angle_estimator.zoa_angles_dict[self._zoa_indices],
                               TOA=self.time_estimator.times_dict[self._toa_indices])
        return estimator
