# Functions

import numpy as np
import scipy.linalg
import scipy.signal


def music(cov: np.ndarray, L: int, n_elements: int, options: np.ndarray):
    # angle_cov is the signal covariance matrix, L is the number of sources, n_elements is the number of n_elements
    # array holds the positions of antenna elements
    # variables are the grid of directions in the azimuth angular domain
    _, V = scipy.linalg.eig(cov)
    Qn = V[:, L:n_elements]
    pspectrum = 1 / scipy.linalg.norm(Qn.conj().T @ options.T, axis=0)
    psindB = np.log10(10 * pspectrum / pspectrum.min())
    DoAsMUSIC, _ = scipy.signal.find_peaks(psindB, height=1.35, distance=1.5)
    return DoAsMUSIC, pspectrum
