from typing import List

import numpy as np

from python_code.estimation.algs import CaponBeamforming


class MultiBandCaponBeamforming(CaponBeamforming):
    """
    The Proposed MultiBand Capon Beamformer.
    """

    def __init__(self, threshs: List[float]):
        super(MultiBandCaponBeamforming, self).__init__(threshs[0])
        self.threshs = threshs

    def run(self, y: List[np.ndarray], basis_vectors: List[np.ndarray], n_elements: List[int],
            second_dim: List[int] = None, third_dim: List[int] = None, use_gpu=False):
        """
        To be completed
        """
        K = len(y)
        norm_values_list = []
        peak_regions = {}
        for k in range(K):
            # compute inverse covariance matrix
            cov = self._compute_cov(n_elements[k], y[k], use_gpu)
            # compute the Capon spectrum values for each basis vector per band
            norm_values = self._compute_capon_spectrum(basis_vectors[k], use_gpu, cov)
            norm_values = norm_values.reshape(-1, second_dim[k])
            norm_values_list.append(norm_values)
        # spectrum refinement step
        low_norm_values = norm_values_list[0]
        low_maximum_ind = np.unravel_index(np.argmax(low_norm_values, axis=None), low_norm_values.shape)
        high_norm_values = norm_values_list[1]
        high_maximum_ind = np.unravel_index(np.argmax(high_norm_values, axis=None), high_norm_values.shape)
        maximum_ind, maximum_value = None, 0
        epsilon_theta, epsilon_tau = 6, 6
        for i in range(low_maximum_ind[0] - epsilon_theta, low_maximum_ind[0] + epsilon_theta + 1):
            for j in range(low_maximum_ind[1] - epsilon_tau, low_maximum_ind[1] + epsilon_tau + 1):
                if high_norm_values[i, j] > maximum_value:
                    maximum_value = high_norm_values[i, j]
                    maximum_ind = [i, j]
        print(high_maximum_ind)
        return np.array([maximum_ind]), low_norm_values
