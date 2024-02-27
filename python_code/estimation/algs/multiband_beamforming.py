from typing import List

import numpy as np

from python_code.estimation.algs import Beamformer

NOISE_FACTOR = 2


class MultiBandBeamformer(Beamformer):
    """
    The Proposed Multi-Band Beamformer.
    """

    def __init__(self):
        super(MultiBandBeamformer, self).__init__()

    def run(self, y: List[np.ndarray], basis_vectors: List[np.ndarray], second_dim: bool = False,
            use_gpu: bool = False):
        K = len(y)
        peak, chosen_k = None, None
        norm_values_list = []
        for k in range(K):
            # compute the spectrum values for each basis vector
            norm_values = self._compute_beamforming_spectrum(basis_vectors[k], use_gpu, y[k])
            maximum_ind = np.array(np.unravel_index(np.argmax(norm_values, axis=None), norm_values.shape))
            norm_values_list.append(norm_values)
            # only if peak is not noisy
            if norm_values[maximum_ind[0], maximum_ind[1]] > NOISE_FACTOR * np.mean(norm_values):
                peak = maximum_ind
                chosen_k = k

        if chosen_k is None:
            peak = maximum_ind
            chosen_k = K-1
        self.k = chosen_k
        return np.array([peak]), norm_values_list[self.k]
