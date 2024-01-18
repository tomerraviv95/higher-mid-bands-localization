import numpy as np
from scipy.optimize import least_squares

from python_code import conf
from python_code.utils.constants import C


def extract_measurements_from_estimations(bs_locs, estimations):
    # extract the toa and aoa measurements for each bs, sorting them such that the LOS is first
    toa_values, aoa_values = [], []
    for i in range(len(bs_locs)):
        cur_toa, cur_aoa = np.array(estimations[i].TOA), np.array(estimations[i].AOA)
        sorting_indices = np.argsort(cur_toa)
        toa_values.append(cur_toa[sorting_indices])
        aoa_values.append(cur_aoa[sorting_indices])
    return toa_values, aoa_values


def optimize_to_estimate_position(bs_locs, estimations, scatterers):
    def cost_func(x):
        costs = []
        # LOS AOA constraints
        for i in range(len(bs_locs)):
            cost = abs(np.arctan2(x[1] - bs_locs[i][1], x[0] - bs_locs[i][0]) - aoa_values[i][0])
            costs.append(cost)
        # LOS TOA constraints
        for i in range(len(bs_locs)):
            cost = abs(np.linalg.norm(x - bs_locs[i]) / C - toa_values[i][0])
            costs.append(cost)
        # scatters constraints
        for i in range(len(bs_locs)):
            # aoa
            for l, nlos_aoa in enumerate(aoa_values[i][1:]):
                cost = abs(
                    np.arctan2(scatterers[l][1] - bs_locs[i][1], scatterers[l][0] - bs_locs[i][0]) - nlos_aoa)
                costs.append(cost)
            # toa
            for l, nlos_toa in enumerate(toa_values[i][1:]):
                cost = abs((np.linalg.norm(bs_locs[i] - scatterers[l]) +
                            np.linalg.norm(conf.ue_pos - scatterers[l])) / C - nlos_toa)
                costs.append(cost)
        return costs

    toa_values, aoa_values = extract_measurements_from_estimations(bs_locs, estimations)
    # LOS computation of location in case of angle and time estimations, or more than one BS/LOS path
    bs_locs = np.array(bs_locs)
    initial_ue_loc = np.array([0, 0])
    est_ue_pos = least_squares(cost_func, initial_ue_loc).x
    return est_ue_pos
