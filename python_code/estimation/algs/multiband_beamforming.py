from typing import List, Tuple

import numpy as np

from python_code.estimation.algs import Beamformer
from python_code.utils.constants import ALG_THRESHOLD


class MultiBandBeamformer(Beamformer):
    """
    The Proposed Multi-Band Beamformer.
    """

    def __init__(self):
        super(MultiBandBeamformer, self).__init__()

    def run(self, y: List[np.ndarray], basis_vectors: List[np.ndarray], second_dim: bool = False,
            use_gpu: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        K = len(y)
        norm_values_list = []
        peaks= []
        peak_values = []
        for k in range(K):
            # compute the spectrum values for sub-band
            norm_values = self._compute_beamforming_spectrum(basis_vectors[k], use_gpu, y[k])
            maximum_ind = np.array(np.unravel_index(np.argmax(norm_values, axis=None), norm_values.shape))
            norm_values_list.append(norm_values)
            peaks.append(maximum_ind)
            peak_values.append(norm_values[maximum_ind[0], maximum_ind[1]])
            print(norm_values[maximum_ind[0], maximum_ind[1]]/np.mean(norm_values))
        max_peak_value = max(peak_values)
        self.k = peak_values.index(max_peak_value)
        peak = peaks[self.k]
        return np.array([peak]), norm_values_list[self.k]
