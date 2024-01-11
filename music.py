# Functions
import numpy as np
import scipy.linalg
import scipy.signal


def array_response_vector(n_elements, var_array):
    return np.exp(-1j * np.pi * var_array) / n_elements ** 0.5


def music(cov, L, n_elements, options,basis_vector):
    # cov is the signal covariance matrix, L is the number of sources, n_elements is the number of n_elements
    # array holds the positions of antenna elements
    # variables are the grid of directions in the azimuth angular domain
    _, V = scipy.linalg.eig(cov)
    Qn = V[:, L:n_elements]
    n_vars = options.shape[0]
    pspectrum = np.zeros(n_vars)
    for i in range(n_vars):
        av = array_response_vector(n_elements, options[i] * basis_vector)
        pspectrum[i] = 1 / scipy.linalg.norm(Qn.conj().T @ av)
    psindB = np.log10(10 * pspectrum / pspectrum.min())
    DoAsMUSIC, _ = scipy.signal.find_peaks(psindB, height=1.35, distance=1.5)
    return DoAsMUSIC, pspectrum
