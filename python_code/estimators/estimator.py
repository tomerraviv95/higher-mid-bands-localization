import math
from collections import namedtuple

import numpy as np

from python_code import conf
from python_code.estimators.beam_sweeper import BeamSweeper
from python_code.estimators.music import MUSIC
from python_code.utils.basis_functions import compute_angle_options, compute_time_options, create_wideband_aoa_mat
from python_code.utils.constants import AlgType

algs = {AlgType.BEAMSWEEPER: BeamSweeper(2), AlgType.MUSIC: MUSIC(1.4)}
ALG_TYPE = AlgType.MUSIC

Estimation = namedtuple("Estimation", ["AOA", "TOA"], defaults=(None,) * 2)


class AngleEstimator:
    def __init__(self):
        self.angles_dict = np.linspace(-np.pi / 2, np.pi / 2, conf.Nb)  # dictionary of spatial frequencies
        self._angle_options = compute_angle_options(self.angles_dict, values=np.arange(conf.Nr))
        self.algorithm = algs[ALG_TYPE]

    def estimate(self, y):
        self._indices, self._spectrum = self.algorithm.run(y=y, basis_vectors=self._angle_options, n_elements=conf.Nr)
        estimator = Estimation(AOA=self.angles_dict[self._indices])
        return estimator


class WidebandAngleEstimator:
    def __init__(self):
        self.angles_dict = np.linspace(-np.pi / 2, np.pi / 2, conf.Nb)  # dictionary of spatial frequencies
        self._angle_options = create_wideband_aoa_mat(self.angles_dict, conf.K, conf.BW, conf.fc, conf.Nr,
                                                      stack_axis=1).reshape(conf.Nr * conf.K, -1).T
        self.algorithm = algs[ALG_TYPE]

    def estimate(self, y):
        self._indices, self._spectrum = self.algorithm.run(y=y, basis_vectors=self._angle_options,
                                                           n_elements=conf.Nr * conf.K)
        return self.angles_dict[self._indices]


class TimeEstimator:
    def __init__(self):
        self.times_dict = np.linspace(0, conf.max_time, conf.T_res)
        self._time_options = compute_time_options(conf.fc, conf.K, conf.BW, values=self.times_dict)
        self.algorithm = algs[ALG_TYPE]

    def estimate(self, y):
        self._indices, self._spectrum = self.algorithm.run(y=np.transpose(y, [1, 0, 2]), n_elements=conf.K,
                                                           basis_vectors=self._time_options)
        estimator = Estimation(TOA=self.times_dict[self._indices])
        return estimator


class AngleTimeEstimator:
    def __init__(self):
        self.angle_estimator = AngleEstimator()
        self.time_estimator = TimeEstimator()
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
