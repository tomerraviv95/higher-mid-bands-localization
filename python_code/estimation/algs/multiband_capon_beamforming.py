from typing import List

import numpy as np

from python_code.estimation.algs import CaponBeamforming


class MultiBandCaponBeamforming(CaponBeamforming):
    """
    The Proposed MultiBand Capon Beamformer.
    """

    def __init__(self, thresh: float):
        super(MultiBandCaponBeamforming, self).__init__(thresh)

    def run(self, y: List[np.ndarray], basis_vectors: List[np.ndarray], n_elements: List[int],
            second_dim: List[int] = None, third_dim: List[int] = None, use_gpu=False):
        """
        To be completed
        """
        norm_values = 0
        for y_single in y:
            # compute inverse covariance matrix
            cov = self._compute_cov(n_elements, y_single)
            # compute the Capon spectrum values for each basis vector per band
            norm_values += self._compute_capon_spectrum(basis_vectors, use_gpu, cov)
        # average the spectrum over the bands
        norm_values /= len(y)
        # finally find the peaks in the spectrum
        return self.find_peaks_in_spectrum(norm_values, second_dim, third_dim)
