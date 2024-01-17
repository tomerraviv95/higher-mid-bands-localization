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


def create_wideband_aoa_mat(aoa_dict, K, BW, fc, Nr, stack_axis):
    wideband_aoa_mat = []
    for k in range(K):
        wideband_angle = aoa_dict * (1 + k * BW / fc)
        wideband_aoa_vec = compute_angle_options(wideband_angle, np.arange(Nr)).T
        wideband_aoa_mat.append(wideband_aoa_vec)
    wideband_aoa_mat = np.concatenate(wideband_aoa_mat, axis=stack_axis)
    return wideband_aoa_mat
