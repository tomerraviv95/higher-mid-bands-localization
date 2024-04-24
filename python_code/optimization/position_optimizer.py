from typing import List, Tuple

import numpy as np
from scipy.optimize import least_squares

from python_code import conf
from python_code.utils.constants import Estimation, BS_ORIENTATION


def extract_measurements_from_estimations(bs_locs: np.ndarray, estimations: List[Estimation]) -> Tuple[
    List[float], List[float], List[np.ndarray]]:
    # extract the toa and aoa measurements for each bs
    toa_values, aoa_values, bs_list = [], [], []
    for i in range(len(bs_locs)):
        toa_values.append(estimations[i].TOA[0])
        aoa_values.append(estimations[i].AOA[0])
        bs_list.append(bs_locs[i])
    return toa_values, aoa_values, bs_list


def optimize_to_estimate_position(bs_locs: np.ndarray, estimations: List[Estimation]) -> np.ndarray:
    def cost_func_2d(ue: np.ndarray) -> List[float]:
        costs = []
        # LOS AOA constraints
        for i in range(len(bs_locs)):
            cost = abs(np.arctan2(ue[1] - bs_locs[i][1], ue[0] - bs_locs[i][0]) - aoa_values[i] - BS_ORIENTATION)
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
    # set the ue location as the bs location at first
    initial_ue_loc = np.array(bs_list[0])
    # estimate the UE using least squares
    res = least_squares(cost_func_2d, initial_ue_loc).x
    est_ue_pos, est_scatterers = res[:2], res[2:].reshape(-1, 2)
    return est_ue_pos
