# Functions
import numpy as np
import scipy.linalg
import scipy.signal

def array_response_vector(n_elements, var):
    array = np.arange(n_elements)
    return np.exp(-1j * np.pi * var * array) / n_elements ** 0.5


def music(CovMat, L, N, Angles):
    # CovMat is the signal covariance matrix, L is the number of sources, N is the number of n_elements
    # array holds the positions of antenna elements
    # Angles are the grid of directions in the azimuth angular domain
    _, V = scipy.linalg.eig(CovMat)
    Qn = V[:, L:N]
    numAngles = Angles.size
    pspectrum = np.zeros(numAngles)
    for i in range(numAngles):
        av = array_response_vector(N, Angles[i])
        pspectrum[i] = 1 / scipy.linalg.norm(Qn.conj().T @ av)
    psindB = np.log10(10 * pspectrum / pspectrum.min())
    DoAsMUSIC, _ = scipy.signal.find_peaks(psindB, height=1.35, distance=1.5)
    return DoAsMUSIC, pspectrum
