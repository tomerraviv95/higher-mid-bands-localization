import numpy as np

from python_code import conf
from python_code.estimation import Estimation
from python_code.estimation.algs import ALG_TYPE, ALGS_DICT
from python_code.utils.basis_functions import compute_time_options
from python_code.utils.constants import MAX_DIST, C


class TimeEstimator2D:
    def __init__(self):
        self.times_dict = np.arange(0, 1 / conf.BW * conf.K, conf.T_res)
        self._time_options = compute_time_options(0, conf.K, conf.BW, values=self.times_dict)
        if self._time_options.shape[0] < self.times_dict.shape[0]:
            self.times_dict = self.times_dict[:self._time_options.shape[0]]
        self.algorithm = ALGS_DICT[ALG_TYPE]

    def estimate(self, y):
        self._indices, self._spectrum, L_hat = self.algorithm.run(y=np.transpose(y, [1, 0, 2]), n_elements=conf.K,
                                                                  basis_vectors=self._time_options)
        estimator = Estimation(TOA=self.times_dict[self._indices])
        return estimator


class TimeEstimator3D:
    def __init__(self):
        self.times_dict = np.linspace(0, MAX_DIST / C, conf.T_res)
        self._time_options = compute_time_options(conf.fc, conf.K, conf.BW, values=self.times_dict)
        self.algorithm = ALGS_DICT[ALG_TYPE]

    def estimate(self, y):
        self._indices, self._spectrum, L_hat = self.algorithm.run(y=np.transpose(y, [2, 1, 0, 3]), n_elements=conf.K,
                                                                  basis_vectors=self._time_options)
        if len(self._indices) > L_hat:
            self._indices = self._indices[:L_hat]
        estimator = Estimation(TOA=self.times_dict[self._indices])
        return estimator
