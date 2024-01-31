import numpy as np

from python_code import conf
from python_code.estimation import Estimation
from python_code.estimation.algs import ALG_TYPE, ALGS_DICT
from python_code.utils.basis_functions import compute_time_options


class TimeEstimator2D:
    def __init__(self):
        self.times_dict = np.arange(0, conf.K / conf.BW, conf.T_res)
        self._time_options = compute_time_options(0, conf.K, conf.BW, values=self.times_dict)
        if self._time_options.shape[0] < self.times_dict.shape[0]:
            self.times_dict = self.times_dict[:self._time_options.shape[0]]
        self.algorithm = ALGS_DICT[ALG_TYPE](2)

    def estimate(self, y):
        self._indices, self._spectrum, L_hat = self.algorithm.run(y=np.transpose(y, [1, 0, 2]), n_elements=conf.K,
                                                                  basis_vectors=self._time_options)
        estimator = Estimation(TOA=self.times_dict[self._indices])
        return estimator


class TimeEstimator3D(TimeEstimator2D):
    def __init__(self):
        super(TimeEstimator3D, self).__init__()

    def estimate(self, y):
        self._indices, self._spectrum, L_hat = self.algorithm.run(y=np.transpose(y, [2, 0, 1, 3]), n_elements=conf.K,
                                                                  basis_vectors=self._time_options)
        estimator = Estimation(TOA=self.times_dict[self._indices])
        return estimator
