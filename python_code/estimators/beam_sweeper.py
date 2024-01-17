import numpy as np
import scipy.linalg
import scipy.signal


class BeamSweeper:
    def __init__(self, thresh: float):
        self.thresh = thresh

    def run(self, y: np.ndarray, basis_vectors: np.ndarray, n_elements: int):
        norm_values = np.linalg.norm(
            basis_vectors.conj() / np.linalg.norm(basis_vectors, axis=1).reshape(-1, 1) @ y.reshape(n_elements, -1),
            axis=1)
        spectrum = np.log10(10 * norm_values / norm_values.min())
        aoa_indices, _ = scipy.signal.find_peaks(spectrum, height=self.thresh)
        return aoa_indices, spectrum
