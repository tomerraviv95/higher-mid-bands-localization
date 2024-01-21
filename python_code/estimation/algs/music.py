# Functions

import numpy as np
import scipy.linalg
import scipy.signal

MUSIC_RESOLUTION = 3


def cluster(evs):
    """
        Estimates multiplicity of smallest eigenvalue.

        @param evs -- The eigenvalues in descending order.

        @returns -- The eigenvalues similar or equal to the smallest eigenvalue.
    """
    # simplest clustering method: with threshold
    threshold = 1
    return evs[np.where(abs(evs) < abs(evs[-1]) + threshold)]


SUB_ARRAY_SIZES = 1000


class MUSIC:
    def __init__(self, thresh):
        self.thresh = thresh

    def run(self, y: np.ndarray, basis_vectors: np.ndarray, n_elements: int, do_one_calc=True):
        # angle_cov is the signal covariance matrix, L is the number of sources, n_elements is the number of n_elements
        # array holds the positions of antenna elements
        # variables are the grid of directions in the azimuth angular domain
        cov = np.cov(y.reshape(n_elements, -1), bias=True)
        eigenvalues, eigenvectors = scipy.linalg.eig(cov)
        smallest_eigenvalue_times = cluster(eigenvalues).shape[0]
        L_hat = n_elements - smallest_eigenvalue_times
        Qn = eigenvectors[:, L_hat:]
        if do_one_calc:
            music_coef = 1 / scipy.linalg.norm(Qn.conj().T @ basis_vectors.T, axis=0)
        else:
            conj_tran_Q = Qn.conj().T
            music_coef = np.zeros(basis_vectors.shape[0])
            batch_size = basis_vectors.shape[0] // SUB_ARRAY_SIZES
            for i in range(0, basis_vectors.shape[0], batch_size):
                music_coef[i:i + batch_size] = 1 / scipy.linalg.norm(conj_tran_Q @ basis_vectors[i:i + batch_size].T,
                                                                     axis=0)
        spectrum = np.log10(10 * music_coef / music_coef.min())
        indices, _ = scipy.signal.find_peaks(spectrum, height=self.thresh, distance=MUSIC_RESOLUTION)
        return indices, spectrum,L_hat
