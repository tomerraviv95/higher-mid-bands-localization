import numpy as np


def array_response_vector(var_array: np.ndarray) -> np.ndarray:
    return np.exp(-2j * np.pi * var_array)


def compute_time_options(fc: float, K: int, BW: float, values: np.ndarray) -> np.ndarray:
    # create the K frequency bins
    time_basis_vector = np.linspace(fc - BW / 2, fc + BW / 2, K)
    # simulate the phase at each frequency bins
    combination = np.dot(values.reshape(-1, 1), time_basis_vector.reshape(1, -1))
    # compute the phase response at each subcarrier
    array_response_combination = array_response_vector(combination)
    # might have duplicates depending on the frequency, BW and number of subcarriers
    # so remove the recurring time basis vectors - assume only the first one is valid
    # and the ones after can only cause recovery errors
    if array_response_combination.shape[0] > 1:
        first_row_duplicates = np.all(np.isclose(array_response_combination, array_response_combination[0]), axis=1)
        if sum(first_row_duplicates) > 1:
            dup_row = np.where(first_row_duplicates)[0][1]
            array_response_combination = array_response_combination[:dup_row]
    return array_response_combination


def compute_angle_options(aoa: np.ndarray, zoa: np.ndarray, values: np.ndarray) -> np.ndarray:
    # create all the combinations for the zoa and aoa degrees
    aoa_zoa_combination = np.kron(aoa, zoa)
    # simulate the degree at each antenna element
    combination = np.dot(aoa_zoa_combination.reshape(-1, 1), values.reshape(1, -1))
    # return the phase at each antenna element in a vector
    return array_response_vector(combination / 2)
