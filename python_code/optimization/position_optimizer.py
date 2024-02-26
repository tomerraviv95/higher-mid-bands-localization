from typing import List, Tuple

import numpy as np
from scipy.optimize import least_squares

from python_code import conf
from python_code.utils.constants import Estimation, L_MAX


def extract_measurements_from_estimations(bs_locs: np.ndarray, estimations: List[Estimation]) -> Tuple[
    List[float], List[float], List[np.ndarray]]:
    # extract the toa and aoa measurements for each bs, sorting them such that the LOS is first
    # if there isn't even a single measurement for the BS then don't add the bs to the list
    toa_values, aoa_values, bs_list = [], [], []
    for i in range(len(bs_locs)):
        if estimations[i].TOA is not None and estimations[i].AOA is not None:
            cur_toa, cur_aoa, cur_power = np.array(estimations[i].TOA), np.array(estimations[i].AOA), np.array(
                estimations[i].POWER)
            toa_power_list = [(ind, toa, -power) for ind, toa, power in zip(range(len(cur_toa)), cur_toa, cur_power)]
            sorted_toa_power_list = sorted(toa_power_list, key=lambda element: (element[1], element[2]))
            min_toa_power_ind = sorted_toa_power_list[0][0]
            toa_values.append(cur_toa[min_toa_power_ind])
            aoa_values.append(cur_aoa[min_toa_power_ind])
            bs_list.append(bs_locs[i])
    return toa_values, aoa_values, bs_list


def optimize_to_estimate_position(bs_locs: np.ndarray, estimations: List[Estimation]) -> np.ndarray:
    def cost_func_2d(x0: np.ndarray) -> List[float]:
        ue, scatterers = x0[:2], x0[2:].reshape(-1, 2)
        costs = []
        # LOS AOA constraints
        for i in range(len(bs_locs)):
            cost = abs(np.arctan2(ue[1] - bs_locs[i][1], ue[0] - bs_locs[i][0]) - aoa_values[i] - conf.orientation)
            costs.append(cost)
        # LOS TOA constraints
        for i in range(len(bs_locs)):
            cost = abs(np.linalg.norm(ue - bs_locs[i]) / conf.medium_speed - toa_values[i])
            costs.append(cost)
        return costs

    # extract the estimated parameters for each bs
    toa_values, aoa_values, bs_list = extract_measurements_from_estimations(bs_locs, estimations)
    print(f'Solving optimization for TOA:{toa_values},AOA:{aoa_values}')
    # LOS computation of location in case of angle and time estimations
    bs_locs = np.array(bs_list)
    initial_ue_loc = np.array([bs_list[0]])
    initial_scatterers = np.zeros([L_MAX, 2])
    x0 = np.concatenate([initial_ue_loc, initial_scatterers]).reshape(-1)
    # estimate the UE using least squares
    res = least_squares(cost_func_2d, x0).x
    est_ue_pos, est_scatterers = res[:2], res[2:].reshape(-1, 2)
    return est_ue_pos
