from typing import List, Union

import numpy as np
import torch

from python_code import DEVICE
from python_code.estimation.algs import ALG_TYPE, ALGS_DICT
from python_code.estimation.parameters.angle import AngleEstimator
from python_code.estimation.parameters.time import TimeEstimator
from python_code.utils.bands_manipulation import Band
from python_code.utils.constants import BandType, Estimation


class AngleTimeEstimator:
    def __init__(self, bands: List[Band]):
        self.angle_estimator = AngleEstimator(bands)
        self.time_estimator = TimeEstimator(bands)
        if self.angle_estimator.multi_band:
            self.mats = [(self.angle_estimator._angle_options[i],self.time_estimator._time_options[i])
                         for i in range(len(bands))]
            self.algorithm = ALGS_DICT[ALG_TYPE][BandType.MULTI]()
        else:
            if torch.cuda.is_available():
                self.mats = [torch.tensor(self.angle_estimator._angle_options).to(DEVICE),
                             torch.tensor(self.time_estimator._time_options).to(DEVICE)]
            else:
                self.mats = [self.angle_estimator._angle_options,
                             self.time_estimator._time_options]
            self.algorithm = ALGS_DICT[ALG_TYPE][BandType.SINGLE]()

    def estimate(self, y: Union[np.ndarray, List[np.ndarray]]) -> Estimation:
        self.indices, self._spectrum = self.algorithm.run(y=y, basis_vectors=self.mats,
                                                          second_dim=True,
                                                          use_gpu=torch.cuda.is_available())
        # if no peaks found - return an empty estimation
        if len(self.indices) == 0:
            return Estimation(AOA=[0], TOA=[0], POWER=[0])
        self._aoa_indices = self.indices[:, 0]
        self._toa_indices = self.indices[:, 1]
        spectrum_powers = self._spectrum[self.indices[:, 0], self.indices[:, 1]]
        AOA = self.angle_estimator.aoa_angles_dict[self._aoa_indices]
        if self.angle_estimator.multi_band:
            TOA = self.time_estimator.times_dict[0][self._toa_indices]
        else:
            TOA = self.time_estimator.times_dict[self._toa_indices]
        estimator = Estimation(AOA=AOA, TOA=TOA, POWER=spectrum_powers)
        return estimator
