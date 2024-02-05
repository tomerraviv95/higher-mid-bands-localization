from typing import Union, List

import numpy as np
import torch

from python_code import DEVICE
from python_code.estimation.parameters_3d.angle import AngleEstimator3D
from python_code.estimation.parameters_3d.time import TimeEstimator3D
from python_code.utils.bands_manipulation import Band
from python_code.utils.constants import Estimation


class AngleTimeEstimator3D:
    def __init__(self, bands: List[Band]):
        self.angle_estimator = AngleEstimator3D(bands)
        self.time_estimator = TimeEstimator3D(bands)
        if self.angle_estimator.multi_band:
            self.n_elements = [self.angle_estimator.n_elements[i] * self.time_estimator.K[i] for i in range(len(bands))]
            mat1s = [self.angle_estimator._angle_options[i].astype(np.complex64) for i in range(len(bands))]
            mat2s = [self.time_estimator._time_options[i].astype(np.complex64) for i in range(len(bands))]
            self.angle_time_options = [self._single_band_constructor(mat1, mat2) for mat1,mat2 in zip(mat1s,mat2s)]
        else:
            self.n_elements = self.angle_estimator.n_elements * self.time_estimator.K
            mat1 = self.angle_estimator._angle_options.astype(np.complex64)
            mat2 = self.time_estimator._time_options.astype(np.complex64)
            self.angle_time_options = self._single_band_constructor(mat1, mat2)

    def _single_band_constructor(self, mat1: np.ndarray, mat2: np.ndarray):
        # if a GPU is available, perform the calculations on it. Note that it is imperative
        # for the 3d case, otherwise expect memory crash on a computer with 16/32 GB RAM.
        if torch.cuda.is_available():
            tensor1 = torch.tensor(mat1, dtype=torch.cfloat).to(DEVICE)
            tensor2 = torch.tensor(mat2, dtype=torch.cfloat).to(DEVICE)

            def angle_time_options_func(batch_ind: int):
                sub_tensor1 = tensor1[batch_ind].reshape(1, -1)
                return torch.kron(sub_tensor1.contiguous(), tensor2.contiguous()), tensor1.shape[0]

            angle_time_options = angle_time_options_func
        else:
            # perform calculations on CPU and RAM
            angle_time_options = np.kron(mat1, mat2)
        return angle_time_options

    def estimate(self, y: Union[np.ndarray, List[np.ndarray]]) -> Estimation:
        second_dim, third_dim = len(self.angle_estimator.zoa_angles_dict), len(self.time_estimator.times_dict)
        self.indices, self._spectrum, _ = self.angle_estimator.algorithm.run(y=y, n_elements=self.n_elements,
                                                                             basis_vectors=self.angle_time_options,
                                                                             second_dim=second_dim,
                                                                             third_dim=third_dim,
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
