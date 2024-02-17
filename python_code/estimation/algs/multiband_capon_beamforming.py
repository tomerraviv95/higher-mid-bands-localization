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
        peaks = {}
        for k in range(K):
            # compute inverse covariance matrix
            cov = self._compute_cov(n_elements[k], y[k], use_gpu)
            # compute the Capon spectrum values for each basis vector per band
            norm_values = self._compute_capon_spectrum(basis_vectors[k], use_gpu, cov)
            norm_values = norm_values.reshape(-1, second_dim[k])
            maximum_ind = np.unravel_index(np.argmax(norm_values, axis=None), norm_values.shape)
            peaks[k] = (maximum_ind, norm_values[maximum_ind])
            norm_values_list.append(norm_values)
        # if the highest frequency is in high confidence - return its peak
        if peaks[1][1] > 1.2:
            print(1)
            return peaks[1][0], norm_values_list[1]
        # otherwise, run the spectrum refinement step
        print(2)
        low_norm_values = norm_values_list[0]
        high_norm_values = norm_values_list[1]
        low_maximum_ind = peaks[0][0]
        maximum_ind, maximum_value = None, 0
        epsilon_theta, epsilon_tau = 3, 0
        for i in range(low_maximum_ind[0] - epsilon_theta, low_maximum_ind[0] + epsilon_theta + 1):
            for j in range(low_maximum_ind[1], low_maximum_ind[1] + epsilon_tau + 1):
                if high_norm_values[i, j] > maximum_value:
                    maximum_value = high_norm_values[i, j]
                    maximum_ind = [i, j]
        return np.array([maximum_ind]), low_norm_values
