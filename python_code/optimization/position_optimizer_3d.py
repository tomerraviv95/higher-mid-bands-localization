from typing import List, Tuple

import numpy as np
from scipy.optimize import least_squares

from python_code.utils.constants import C, Estimation

MAX_L = 5


def extract_measurements_from_estimations(bs_locs: np.ndarray, estimations: List[Estimation]) -> Tuple[
    List[float], List[float], List[float], List[np.ndarray]]:
    # extract the toa, aoa and zoa measurements for each bs, sorting them such that the LOS is first
    # if there isn't even a single measurement for the BS then don't add the bs to the list
    aoa_values, zoa_values, toa_values, bs_list = [], [], [], []
    for i in range(len(bs_locs)):
        if estimations[i].TOA is not None and estimations[i].AOA is not None:
            cur_aoa, cur_zoa = np.array(estimations[i].AOA), np.array(estimations[i].ZOA)
            cur_toa = np.array(estimations[i].TOA)
            sorting_indices = np.argsort(cur_toa)
            aoa_values.append(cur_aoa[sorting_indices])
            zoa_values.append(cur_zoa[sorting_indices])
            toa_values.append(cur_toa[sorting_indices])
            bs_list.append(bs_locs[i])
    return aoa_values, zoa_values, toa_values, bs_list


def optimize_to_estimate_position_3d(bs_locs: np.ndarray, estimations: List[Estimation]) -> np.ndarray:
    def cost_func_3d(x0: np.ndarray) -> List[float]:
        ue, scatterers = x0[:3], x0[3:].reshape(-1, 3)
        costs = []
        # LOS AOA constraints
        for i in range(len(bs_locs)):
            cost = abs(np.arctan2(ue[1] - bs_locs[i][1], ue[0] - bs_locs[i][0]) - aoa_values[i][0])
            costs.append(cost)
        # LOS ZOA constraints
        for i in range(len(bs_locs)):
            cost = abs(np.arctan2(np.linalg.norm(bs_locs[i][:2] - ue[:2]), bs_locs[i][2] - ue[2]) - zoa_values[i][0])
            costs.append(cost)
        # LOS TOA constraints
        for i in range(len(bs_locs)):
            cost = abs(np.linalg.norm(ue - bs_locs[i]) / C - toa_values[i][0])
            costs.append(cost)
        return costs

    # extract the estimated parameters_2d for each bs
    aoa_values, zoa_values, toa_values, bs_list = extract_measurements_from_estimations(bs_locs, estimations)
    # LOS computation of location in case of angle and time estimations
    bs_locs = np.array(bs_list)
    initial_ue_loc = np.array([[0, 0, 0]])
    initial_scatterers = np.zeros([MAX_L, 3])
    x0 = np.concatenate([initial_ue_loc, initial_scatterers]).reshape(-1)
    # estimate the UE using least squares
    res = least_squares(cost_func_3d, x0).x
    est_ue_pos, est_scatterers = res[:3], res[3:].reshape(-1, 3)
    return est_ue_pos
