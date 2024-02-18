from typing import List

import numpy as np

from python_code.estimation.algs import CaponBeamforming
from python_code.utils.constants import MAX_COMPONENTS


class MultiBandCaponBeamforming(CaponBeamforming):
    """
    The Proposed MultiBand Capon Beamformer.
    """

    def __init__(self, thresh: float):
        super(MultiBandCaponBeamforming, self).__init__(thresh)
        self.thresh = thresh

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
            labeled, ncomponents = self.label_spectrum_by_peaks(norm_values)
            if ncomponents == 0:
                maximum_ind = np.unravel_index(np.argmax(norm_values, axis=None), norm_values.shape)
                min_toa_components = 0
            else:
                # each group is tuple (toa,max_power,max_ind)
                s_groups = self.compute_peaks_groups(labeled, ncomponents, norm_values)
                # minimal TOA, maximum power peak
                maximum_ind = s_groups[0][2]
                min_toa_components = len(s_groups)
            peaks[min_toa_components] = (maximum_ind, k)
            norm_values_list.append(norm_values)
        # run the greedy peak selection phase
        for n_components in range(1, MAX_COMPONENTS + 1):
            if n_components in peaks.keys():
                maximum_ind, k = peaks[n_components]
                return np.array([maximum_ind]), norm_values_list[k], k
        maximum_ind, k = peaks[0]
        return np.array([maximum_ind]), norm_values_list[k], k
