from typing import List

import numpy as np
from scipy.ndimage import label

from python_code.estimation.algs import CaponBeamforming

MAX_COMPONENTS = 5


def count_components(norm_values: np.ndarray) -> int:
    max_indices = np.argsort(norm_values, axis=None)[::-1][:MAX_COMPONENTS]
    indices = np.array(np.unravel_index(max_indices, norm_values.shape)).T
    image = np.zeros_like(norm_values)
    for ind in indices:
        image[ind[0], ind[1]] = 1
    _, ncomponents = label(image, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int))
    return ncomponents


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
            maximum_ind = np.unravel_index(np.argmax(norm_values, axis=None), norm_values.shape)
            n_components = count_components(norm_values)
            if norm_values[maximum_ind] > self.thresh:
                peaks[n_components] = (maximum_ind, k)
            norm_values_list.append(norm_values)
        print(peaks)
        # otherwise, run the spectrum refinement step
        for n_components in range(1, MAX_COMPONENTS):
            if n_components in peaks.keys():
                maximum_ind, k = peaks[n_components]
                print(f"Chosen: {str(6) if k==0 else str(24)}")
                return np.array([maximum_ind]), norm_values_list[k], k
