import numpy as np
import scipy.signal
from findpeaks import findpeaks

fp = findpeaks(lookahead=1)


class CaponBeamforming:
    def __init__(self, thresh: float):
        self.thresh = thresh

    def run(self, y: np.ndarray, basis_vectors: np.ndarray, n_elements: int):
        cov = np.cov(y.reshape(n_elements, -1), bias=True)
        inv_cov = np.linalg.inv(cov)
        inv_cov = inv_cov / np.linalg.norm(inv_cov)
        norm_values = np.linalg.norm((basis_vectors.conj() @ inv_cov) * basis_vectors, axis=1)
        norm_values = 1 / norm_values
        indices, _ = scipy.signal.find_peaks(norm_values, height=self.thresh)
        return indices, norm_values, len(indices)
