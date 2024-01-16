# Functions

import numpy as np
import scipy.linalg
import scipy.signal

def cluster(evs):
    """
        Estimates multiplicity of smallest eigenvalue.

        @param evs -- The eigenvalues in descending order.

        @returns -- The eigenvalues similar or equal to the smallest eigenvalue.
    """
    # simplest clustering method: with threshold
    threshold = 0.1
    return evs[np.where(abs(evs) < abs(evs[-1]) + threshold)]

def music(cov: np.ndarray, n_elements: int, options: np.ndarray):
    # angle_cov is the signal covariance matrix, L is the number of sources, n_elements is the number of n_elements
    # array holds the positions of antenna elements
    # variables are the grid of directions in the azimuth angular domain
    eigenvalues, eigenvectors = scipy.linalg.eig(cov)
    smallest_eigenvalue_times = cluster(eigenvalues).shape[0]
    L_hat = n_elements - smallest_eigenvalue_times
    Qn = eigenvectors[:, L_hat:]
    music_coef = 1 / scipy.linalg.norm(Qn.conj().T @ options.T, axis=0)
    spectrum = np.log10(10 * music_coef / music_coef.min())
    aoa_indices, _ = scipy.signal.find_peaks(spectrum, height=1.15)
    top_aoa_indices = aoa_indices[np.argsort(spectrum[aoa_indices])[-L_hat - 1:]]
    return top_aoa_indices, spectrum
