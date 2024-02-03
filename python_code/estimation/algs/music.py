# Functions

import numpy as np
import scipy.linalg
import scipy.signal

MUSIC_RESOLUTION = 3


def cluster(evs: np.ndarray) -> int:
    """
        Estimates multiplicity of smallest eigenvalue.

        @param evs -- The eigenvalues in descending order.

        @returns -- The eigenvalues similar or equal to the smallest eigenvalue.
    """
    # simplest clustering method: with threshold
    threshold = 0.1
    return evs[np.where(abs(evs) / abs(evs)[0] < threshold)].shape[0]


class MUSIC:
    """
    The implementation of MUSIC algorithm
    Supported for only 1d signals
    For extensions for 2d or 3d please refer to the Capon beamformer class
    """

    def __init__(self, thresh):
        self.thresh = thresh

    def run(self, y: np.ndarray, basis_vectors: np.ndarray, n_elements: int):
        # angle_cov is the signal covariance matrix, L is the number of sources, n_elements is the number of n_elements
        # array holds the positions of antenna elements
        # variables are the grid of directions in the azimuth angular domain
        cov = np.cov(y.reshape(n_elements, -1), bias=True)
        eigenvalues, eigenvectors = scipy.linalg.eig(cov)
        smallest_eigenvalue_times = cluster(eigenvalues)
        L_hat = n_elements - smallest_eigenvalue_times
        Qn = eigenvectors[:, L_hat:]
        music_coef = 1 / scipy.linalg.norm(Qn.conj().T @ basis_vectors.T, axis=0)
        spectrum = np.log10(10 * music_coef / music_coef.min())
        indices, _ = scipy.signal.find_peaks(spectrum, height=self.thresh, distance=MUSIC_RESOLUTION)
        return indices, spectrum, L_hat
