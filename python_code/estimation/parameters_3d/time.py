from typing import List

import numpy as np

from python_code.estimation.parameters_2d.time import TimeEstimator2D
from python_code.utils.bands_manipulation import Band
from python_code.utils.constants import Estimation


class TimeEstimator3D(TimeEstimator2D):
    def __init__(self, bands: List[Band]):
        super(TimeEstimator3D, self).__init__(bands)

    def estimate(self, y: np.ndarray) -> Estimation:
        self._indices, self._spectrum, _ = self.algorithm.run(y=np.transpose(y, [2, 0, 1, 3]), n_elements=self.K,
                                                              basis_vectors=self._time_options)
        estimator = Estimation(TOA=self.times_dict[self._indices])
        return estimator