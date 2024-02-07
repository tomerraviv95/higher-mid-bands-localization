from typing import List

import numpy as np
from scipy.ndimage.measurements import label

from python_code.estimation.algs import CaponBeamforming


class MultiBandCaponBeamforming(CaponBeamforming):
    """
    The Proposed MultiBand Capon Beamformer.
    """

    def __init__(self, thresh: float):
        super(MultiBandCaponBeamforming, self).__init__(thresh)

    def run(self, y: List[np.ndarray], basis_vectors: List[np.ndarray], n_elements: List[int],
            second_dim: int = None, third_dim: int = None, use_gpu=False):
        """
        To be completed
        """
        norm_values_list = []
        for i in range(len(y)):
            # compute inverse covariance matrix
            cov = self._compute_cov(n_elements[i], y[i])
            # compute the Capon spectrum values for each basis vector per band
            norm_values = self._compute_capon_spectrum(basis_vectors[i], use_gpu, cov)
            norm_values_list.append(norm_values.reshape(-1, second_dim))
        low_norm_values, high_norm_values = norm_values_list[0], norm_values_list[1]
        # replace the peaks in the low spectrum by peaks of higher spectrum, if the peak at lower spectrum
        # is also a peak at the higher spectrum. Otherwise, leave the lower spectrum peak.
        labeled, ncomponents = label(low_norm_values > self.thresh, structure=np.ones((3, 3), dtype=int))
        for component in range(1, ncomponents + 1):
            # get the region indices
            component_indices = np.array(np.where(labeled == component)).T
            # find if at least one index in the higher frequency is peak
            peak_found = False
            for component_indx in component_indices:
                cur_comp = self.get_current_component(norm_values_list[1], component_indx, third_dim)
                if cur_comp > self.thresh:
                    peak_found = True
            # only if it is peak in higher frequency - replace the spectrum with the more accurate values
            if peak_found:
                for component_indx in component_indices:
                    low_norm_values[component_indx[0]][component_indx[1]] = \
                        high_norm_values[component_indx[0]][component_indx[1]]
        # finally find the peaks in the spectrum
        return self.find_peaks_in_spectrum(low_norm_values.reshape(-1), second_dim, third_dim)
