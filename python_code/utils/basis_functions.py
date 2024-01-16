import numpy as np


def array_response_vector(n_elements, var_array):
    return np.exp(-1j * np.pi * var_array) / n_elements ** 0.5


def compute_time_options(fc, K, BW, values):
    time_basis_vector = 2 * (fc + np.arange(K) * BW / K)
    combination = np.dot(values.reshape(-1, 1), time_basis_vector.reshape(1, -1))
    return array_response_vector(len(values), combination)


def compute_angle_options(aa, values):
    combination = np.dot(np.sin(aa).reshape(-1, 1), values.reshape(1, -1))
    return array_response_vector(len(values), combination)
