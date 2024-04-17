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
        peak, chosen_k = None, None
        norm_values_list = []
        for k in range(K):
            # compute the spectrum values for sub-band
            norm_values = self._compute_beamforming_spectrum(basis_vectors[k], use_gpu, y[k])
            maximum_ind = np.array(np.unravel_index(np.argmax(norm_values, axis=None), norm_values.shape))
            norm_values_list.append(norm_values)
            # only if peak is above noise level
            print(k, norm_values[maximum_ind[0], maximum_ind[1]] / np.mean(norm_values))
            if norm_values[maximum_ind[0], maximum_ind[1]] > ALG_THRESHOLD * np.mean(norm_values):
                peak = maximum_ind
                chosen_k = k
        # if all are below the noise level - choose the last sub-band
        if chosen_k is None:
            peak = maximum_ind
            chosen_k = K - 1
        self.k = chosen_k
        return np.array([peak]), norm_values_list[self.k]
