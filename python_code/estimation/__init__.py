from collections import namedtuple

import numpy as np

from python_code.utils.bands_manipulation import Band

Estimation = namedtuple("Estimation", ["AOA", "TOA", "ZOA"], defaults=(None,) * 3)


class Estimator:
    def __init__(self, band: Band):
        pass

    def estimate(self, y: np.ndarray) -> Estimation:
        pass
