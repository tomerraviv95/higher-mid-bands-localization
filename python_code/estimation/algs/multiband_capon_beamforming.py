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
            second_dim: List[int] = None, third_dim: List[int] = None, use_gpu=False):
        """
        To be completed
        """
        K=len(y)
        norm_values_list = []
        peaks = {}
        for k in range(K):
            # compute inverse covariance matrix
            cov = self._compute_cov(n_elements[k], y[k], use_gpu)
            # compute the Capon spectrum values for each basis vector per band
            norm_values = self._compute_capon_spectrum(basis_vectors[k], use_gpu, cov).reshape(-1, second_dim[k])
            labeled, ncomponents = label(norm_values > self.thresh, structure=np.ones((3, 3), dtype=int))
            peaks[k] = [(np.array(np.where(labeled == component)).T, k) for component in range(1, ncomponents + 1)]
            norm_values_list.append(norm_values)
        # spectrum refinement step
        # replace the peaks in the low spectrum by peaks of higher spectrum
        refined_peaks = peaks[0]
        k = 1
        while k < K:
            next_refined_peaks = []
            for low_res_peak, low_res_k in refined_peaks:
                intersected_peaks = 0
                for high_res_peak, cur_k in peaks[k]:
                    low_res_peak_set = set([tuple(ele) for ele in low_res_peak])
                    high_res_peak_set = set([tuple(ele) for ele in high_res_peak])
                    intersection = low_res_peak_set.intersection(high_res_peak_set)
                    if len(intersection) > 0:
                        next_refined_peaks.append((high_res_peak, cur_k))
                        intersected_peaks += 1
                if intersected_peaks == 0:
                    next_refined_peaks.append((low_res_peak, low_res_k))
            refined_peaks = next_refined_peaks
            k += 1
        high_spectrum = norm_values_list[-1]
        for refined_peak,k in refined_peaks:
            for peak_ind in refined_peak:
                i, j = peak_ind[0], peak_ind[1]
                high_spectrum[i][j] = norm_values_list[k][i][j]
        # finally find the peaks in the spectrum
        return self.find_peaks_in_spectrum(high_spectrum.reshape(-1), second_dim[0],third_dim)
