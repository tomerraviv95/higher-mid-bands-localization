import numpy as np
from scipy.optimize import least_squares

from python_code import conf
from python_code.utils.constants import C

MAX_L = 5


def extract_measurements_from_estimations(bs_locs, estimations):
    # extract the toa and aoa measurements for each bs, sorting them such that the LOS is first
    toa_values, aoa_values = [], []
    for i in range(len(bs_locs)):
        cur_toa, cur_aoa = np.array(estimations[i].TOA), np.array(estimations[i].AOA)
        sorting_indices = np.argsort(cur_toa)
        toa_values.append(cur_toa[sorting_indices])
        aoa_values.append(cur_aoa[sorting_indices])
    return toa_values, aoa_values


def optimize_to_estimate_position_2d(bs_locs, estimations):
    def cost_func_2d(x0):
        ue, scatterers = x0[:2], x0[2:].reshape(-1, 2)
        costs = []
        # LOS AOA constraints
        for i in range(len(bs_locs)):
            cost = abs(np.arctan2(ue[1] - bs_locs[i][1], ue[0] - bs_locs[i][0]) - aoa_values[i][0])
            costs.append(cost)
        # LOS TOA constraints
        for i in range(len(bs_locs)):
            cost = abs(np.linalg.norm(ue - bs_locs[i]) / C - toa_values[i][0])
            costs.append(cost)
        # scatters constraints
        for i in range(len(bs_locs)):
            # aoa
            for l, nlos_aoa in enumerate(aoa_values[i][1:]):
                if l >= MAX_L:
                    break
                cost = abs(
                    np.arctan2(scatterers[l][1] - bs_locs[i][1], scatterers[l][0] - bs_locs[i][0]) - nlos_aoa)
                costs.append(cost)
            # toa
            for l, nlos_toa in enumerate(toa_values[i][1:]):
                if l >= MAX_L:
                    break
                cost = abs((np.linalg.norm(bs_locs[i] - scatterers[l]) +
                            np.linalg.norm(conf.ue_pos - scatterers[l])) / C - nlos_toa)
                costs.append(cost)
        return costs

    toa_values, aoa_values = extract_measurements_from_estimations(bs_locs, estimations)
    # LOS computation of location in case of angle and time estimations, or more than one BS/LOS path
    bs_locs = np.array(bs_locs)
    initial_ue_loc = np.array([[0, 0]])
    initial_scatterers = np.zeros([MAX_L, 2])
    x0 = np.concatenate([initial_ue_loc, initial_scatterers]).reshape(-1)
    res = least_squares(cost_func_2d, x0).x
    est_ue_pos, est_scatterers = res[:2], res[2:].reshape(-1, 2)
    return est_ue_pos