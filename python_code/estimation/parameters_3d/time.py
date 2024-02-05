from typing import List, Union

import numpy as np

from python_code.estimation.parameters_2d.time import TimeEstimator2D
from python_code.utils.bands_manipulation import Band
from python_code.utils.constants import Estimation


class TimeEstimator3D(TimeEstimator2D):
    def __init__(self, bands: List[Band]):
        super(TimeEstimator3D, self).__init__(bands)

    def estimate(self, y: Union[np.ndarray, List[np.ndarray]]) -> Estimation:
        if self.multi_band:
            y = [np.transpose(y_single, [2, 0, 1, 3]) for y_single in y]
        else:
            y = np.transpose(y, [2, 0, 1, 3])
        self._indices, self._spectrum, _ = self.algorithm.run(y=y, n_elements=self.K,
                                                              basis_vectors=self._time_options)
        estimator = Estimation(TOA=self.times_dict[self._indices])
        return estimator