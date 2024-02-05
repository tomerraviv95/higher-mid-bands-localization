from typing import Union, List

import numpy as np
import torch

from python_code import DEVICE
from python_code.estimation.algs import ALG_TYPE, ALGS_DICT
from python_code.estimation.parameters_3d import coef_per_frequencies_dict
from python_code.estimation.parameters_3d.angle import AngleEstimator3D
from python_code.estimation.parameters_3d.time import TimeEstimator3D
from python_code.utils.bands_manipulation import Band
from python_code.utils.constants import Estimation


class AngleTimeEstimator3D:
    def __init__(self, bands: List[Band]):
        self.angle_estimator = AngleEstimator3D(bands)
        self.time_estimator = TimeEstimator3D(bands)
        if self.angle_estimator.multi_band:
            self._multiband_constructor(bands)
        else:
            self._single_band_constructor(bands)

    def _single_band_constructor(self, bands):
        self.n_elements = self.angle_estimator.n_elements * self.time_estimator.K
        mat1 = self.angle_estimator._angle_options.astype(np.complex64)
        mat2 = self.time_estimator._time_options.astype(np.complex64)
        # if a GPU is available, perform the calculations on it. Note that it is imperative
        # for the 3d case, otherwise expect memory crash on a computer with 16/32 GB RAM.
        if torch.cuda.is_available():
            tensor1 = torch.tensor(mat1, dtype=torch.cfloat).to(DEVICE)
            tensor2 = torch.tensor(mat2, dtype=torch.cfloat).to(DEVICE)

            def angle_time_options_func(batch_ind: int):
                sub_tensor1 = tensor1[batch_ind].reshape(1, -1)
                return torch.kron(sub_tensor1.contiguous(), tensor2.contiguous()), tensor1.shape[0]

            self.angle_time_options = angle_time_options_func
        else:
            # perform calculations on CPU and RAM
            self.angle_time_options = np.kron(mat1, mat2)
        self.batches = self.time_estimator._time_options.shape[0]

    def estimate(self, y: Union[np.ndarray, List[np.ndarray]]) -> Estimation:
        self.indices, self._spectrum, _ = self.angle_estimator.algorithm.run(y=y, n_elements=self.n_elements,
                                                                             basis_vectors=self.angle_time_options,
                                                                             second_dim=len(
                                                                                 self.angle_estimator.zoa_angles_dict),
                                                                             third_dim=len(
                                                                                 self.time_estimator.times_dict),
                                                                             use_gpu=torch.cuda.is_available())
        # if no peaks found - return an empty estimation
        if len(self.indices) == 0:
            return Estimation()
        self._aoa_indices = self.indices[:, 0]
        self._zoa_indices = self.indices[:, 1]
        self._toa_indices = self.indices[:, 2]
        estimator = Estimation(AOA=self.angle_estimator.aoa_angles_dict[self._aoa_indices],
                               ZOA=self.angle_estimator.zoa_angles_dict[self._zoa_indices],
                               TOA=self.time_estimator.times_dict[self._toa_indices])
        return estimator
