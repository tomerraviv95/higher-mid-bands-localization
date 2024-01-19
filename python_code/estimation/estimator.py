import math

import numpy as np

from python_code import conf
from python_code.estimation import ALG_TYPE, Estimation
from python_code.estimation.angle import AngleEstimator3D
from python_code.estimation.time import TimeEstimator2D


class AngleTimeEstimator:
    def __init__(self, algs=None):
        self.angle_estimator = AngleEstimator3D()
        self.time_estimator = TimeEstimator2D()
        self.angle_time_options = np.kron(self.angle_estimator._angle_options, self.time_estimator._time_options)
        self.algorithm = algs[ALG_TYPE]

    def estimate(self, y):
        indices, self._spectrum = self.algorithm.run(y=y, n_elements=conf.Nr * conf.K,
                                                     basis_vectors=self.angle_time_options)
        # filter nearby detected peaks
        aoa_toa_set = self.filter_peaks(indices)
        aoa_list, toa_list = zip(*aoa_toa_set)
        estimator = Estimation(AOA=aoa_list, TOA=toa_list)
        return estimator

    def filter_peaks(self, indices):
        aoa_indices = indices // conf.T_res
        toa_indices = indices % conf.T_res
        aoa_toa_set = set()
        for unique_toa_ind in np.unique(toa_indices):
            toa = self.time_estimator.times_dict[unique_toa_ind]
            avg_aoa_ind = int(np.mean(aoa_indices[toa_indices == unique_toa_ind]))
            aoa = self.angle_estimator.angles_dict[avg_aoa_ind]
            if not self.set_contains(aoa_toa_set, (aoa, toa)):
                aoa_toa_set.add((aoa, toa))
        return aoa_toa_set

    @staticmethod
    def set_contains(aoa_toa_set, aoa_toa_tuple):
        for c_aoa, c_toa in aoa_toa_set:
            if abs(aoa_toa_tuple[0] - c_aoa) < 5 * math.pi / 180 and abs(aoa_toa_tuple[1] - c_toa) < 0.02:
                return True
        return False
