import numpy as np

from python_code import conf
from python_code.estimation import Estimation
from python_code.estimation.algs import ALG_TYPE, ALGS_DICT
from python_code.utils.basis_functions import compute_time_options


class TimeEstimator2D:
    def __init__(self):
        self.times_dict = np.linspace(0, conf.max_time, conf.T_res)
        self._time_options = compute_time_options(conf.fc, conf.K, conf.BW, values=self.times_dict)
        self.algorithm = ALGS_DICT[ALG_TYPE]

    def estimate(self, y):
        self._indices, self._spectrum = self.algorithm.run(y=np.transpose(y, [1, 0, 2]), n_elements=conf.K,
                                                           basis_vectors=self._time_options)
        estimator = Estimation(TOA=self.times_dict[self._indices])
        return estimator


class TimeEstimator3D:
    def __init__(self):
        self.times_dict = np.linspace(0, conf.max_time, conf.T_res)
        # self.times_dict = np.linspace(1, conf.max_time, conf.T_res)
        self._time_options = compute_time_options(conf.fc, conf.K, conf.BW, values=self.times_dict)
        self.algorithm = ALGS_DICT[ALG_TYPE]

    def estimate(self, y):
        self._indices, self._spectrum = self.algorithm.run(y=np.transpose(y, [2, 1, 0, 3]), n_elements=conf.K,
                                                           basis_vectors=self._time_options)
        estimator = Estimation(TOA=self.times_dict[self._indices])
        return estimator
