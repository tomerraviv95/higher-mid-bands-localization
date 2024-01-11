# Functions
import itertools

import numpy as np
import scipy.linalg
import scipy.signal


class VarsOptions:
    def __init__(self, vars_num):
        self.vars_nums = vars_num
        self.options = [[] for _ in range(vars_num)]

    def set_options(self, var_ind, values):
        if var_ind >= self.vars_nums:
            raise ValueError("Val index too high")
        self.options[var_ind] = values

    def get_options(self):
        return list(itertools.product(*self.options))


def array_response_vector(n_elements, var_array):
    return np.exp(-1j * np.pi * var_array) / n_elements ** 0.5


def compute_time_options(fc, K, BW, values):
    time_basis_vector = 2 * (fc + np.arange(K) * BW / K)
    combination = np.dot(values.reshape(-1, 1), time_basis_vector.reshape(1, -1))
    return array_response_vector(len(values), combination)


def compute_angle_options(aa, values):
    combination = np.dot(np.sin(aa).reshape(-1, 1), values.reshape(1, -1))
    return array_response_vector(len(values), combination)


def music(cov, L, n_elements: int, options: np.ndarray):
    # angle_cov is the signal covariance matrix, L is the number of sources, n_elements is the number of n_elements
    # array holds the positions of antenna elements
    # variables are the grid of directions in the azimuth angular domain
    _, V = scipy.linalg.eig(cov)
    Qn = V[:, L:n_elements]
    pspectrum = 1 / scipy.linalg.norm(Qn.conj().T @ options.T, axis=0)
    psindB = np.log10(10 * pspectrum / pspectrum.min())
    DoAsMUSIC, _ = scipy.signal.find_peaks(psindB, height=1.35, distance=1.5)
    return DoAsMUSIC, pspectrum
