import numpy as np


def array_response_vector(var_array: np.ndarray) -> np.ndarray:
    return np.exp(-2j * np.pi * var_array)


def compute_time_options(fc: float, K: int, BW: float, values: np.ndarray) -> np.ndarray:
    time_basis_vector = np.linspace(fc - BW / 2, fc + BW / 2, K)
    combination = np.dot(values.reshape(-1, 1), time_basis_vector.reshape(1, -1))
    array_response_combination = array_response_vector(combination)
    if array_response_combination.shape[0] > 1:
        first_row_duplicates = np.all(np.isclose(array_response_combination, array_response_combination[0]), axis=1)
        if sum(first_row_duplicates) > 1:
            dup_row = np.where(first_row_duplicates)[0][1]
            array_response_combination = array_response_combination[:dup_row]
    return array_response_combination


def compute_angle_options(aoa: np.ndarray, zoa: np.ndarray, values: np.ndarray):
    aoa_zoa_combination = np.kron(aoa, zoa)
    combination = np.dot(aoa_zoa_combination.reshape(-1, 1), values.reshape(1, -1))
    return array_response_vector(combination / 2)


def create_wideband_aoa_mat(aoa_dict, K, BW, fc, Nr, stack_axis):
    wideband_aoa_mat = []
    for k in range(K):
        wideband_angle = aoa_dict * (1 + k * BW / fc)
        wideband_aoa_vec = compute_angle_options(wideband_angle, np.arange(Nr)).T
        wideband_aoa_mat.append(wideband_aoa_vec)
    wideband_aoa_mat = np.concatenate(wideband_aoa_mat, axis=stack_axis)
    return wideband_aoa_mat
