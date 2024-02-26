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
        """
        To be completed
        """
        K = len(y)
        peaks = []
        for k in range(K):
            # compute the spectrum values for each basis vector
            norm_values = self._compute_beamforming_spectrum(basis_vectors[k], use_gpu, y[k])
            maximum_ind = np.array(np.unravel_index(np.argmax(norm_values, axis=None), norm_values.shape))
            # only if peak is not noisy
            if norm_values[maximum_ind[0], maximum_ind[1]] > NOISE_FACTOR * np.mean(norm_values):
                aoa, toa = maximum_ind[0], maximum_ind[1]
                peaks.append((aoa, toa, k))
        # sort by sub-band index
        s_peaks = sorted(peaks, key=lambda x: x[2])
        # choose peak as the toa of the smallest subband, and the aoa of the highest subband
        max_aoa = s_peaks[-1][0]
        max_toa = s_peaks[0][1]
        return np.array([[max_aoa, max_toa]]), norm_values
