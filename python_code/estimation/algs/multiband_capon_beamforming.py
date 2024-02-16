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
            peak_regions[k] = self._find_peaks_in_spectrum(norm_values, self.threshs[k], second_dim[k])
            norm_values_list.append(norm_values)
        # spectrum refinement step
        # replace the peaks in the low spectrum by peaks of higher spectrum
        refined_peaks = {0: peak_regions[0]}
        k = 1
        while k < K:
            next_refined_peaks = {}
            for low_res_peak, low_res_region in refined_peaks[k - 1].items():
                intersected_peaks = 0
                for high_res_peak, high_res_region in peak_regions[k].items():
                    low_res_peak_set = set(low_res_region)
                    high_res_peak_set = set(high_res_region)
                    intersection = low_res_peak_set.intersection(high_res_peak_set)
                    if len(intersection) > 0:
                        next_refined_peaks[high_res_peak] = high_res_region
                        intersected_peaks += 1
                if intersected_peaks == 0:
                    next_refined_peaks[low_res_peak] = low_res_region
            refined_peaks[k] = next_refined_peaks
            k += 1
        # finally find the peaks in the spectrum
        return np.array(list(refined_peaks[K - 1].keys())), norm_values_list[0]
