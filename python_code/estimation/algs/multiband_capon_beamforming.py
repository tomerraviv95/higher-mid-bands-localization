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
        # otherwise, run the spectrum refinement step
        low_norm_values = norm_values_list[0]
        maximum_inds = np.unravel_index(np.argsort(low_norm_values, axis=None)[::-1][:5], norm_values.shape)
        print(maximum_inds, norm_values[maximum_inds[0], maximum_inds[1]])
        high_norm_values = norm_values_list[1]
        epsilon_theta, epsilon_tau = 5, 5
        for local_max in zip(list(maximum_inds)):
            maximum_ind, maximum_value = None, 0
            for i in range(local_max[0] - epsilon_theta, local_max[0] + epsilon_theta + 1):
                for j in range(local_max[1], local_max[1] + epsilon_tau + 1):
                    if high_norm_values[i, j] > maximum_value:
                        maximum_value = high_norm_values[i, j]
                        maximum_ind = [i, j]
            print(local_max,maximum_ind, maximum_value)
        return np.array([maximum_ind]), low_norm_values
