import numpy as np


def array_response_vector(var_array):
    return np.exp(-2j * np.pi * var_array)


def compute_time_options(fc, K, BW, values):
    time_basis_vector = fc + np.arange(K) * BW / K
    combination = np.dot(values.reshape(-1, 1), time_basis_vector.reshape(1, -1))
    return array_response_vector(combination)


def compute_angle_options(angle_value, values):
    combination = np.dot(np.sin(angle_value).reshape(-1, 1) / 2, values.reshape(1, -1))
    return array_response_vector(combination)
