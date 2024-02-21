from typing import List

import numpy as np

from python_code.estimation.algs import CaponBeamforming


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
        peaks = []
        for k in range(K):
            # compute inverse covariance matrix
            cov = self._compute_cov(n_elements[k], y[k], use_gpu)
            # compute the Capon spectrum values for each basis vector per band
            norm_values = self._compute_capon_spectrum(basis_vectors[k], use_gpu, cov)
            norm_values = norm_values.reshape(-1, second_dim[k])
            maximum_ind = np.unravel_index(np.argmax(norm_values, axis=None), norm_values.shape)
            norm_values_list.append(norm_values)
            if norm_values[maximum_ind[0], maximum_ind[1]] > self.thresh:
                min_toa = maximum_ind[1]
                peaks.append((maximum_ind, min_toa, k))
        # sort by min toa, then band
        s_peaks = sorted(peaks, key=lambda x: (x[1], -x[2]))
        global_peak = s_peaks[0][0]
        k = s_peaks[0][2]
        return np.array([global_peak]), norm_values_list[k], k
