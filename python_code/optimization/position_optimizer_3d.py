import numpy as np
from scipy.optimize import least_squares

from python_code import conf
from python_code.utils.constants import C

MAX_L = 5


def extract_measurements_from_estimations(bs_locs, estimations):
    # extract the toa and aoa measurements for each bs, sorting them such that the LOS is first
    aoa_values, zoa_values, toa_values, = [], [], []
    for i in range(len(bs_locs)):
        cur_aoa, cur_zoa, cur_toa = np.array(estimations[i].AOA), np.array(estimations[i].ZOA), \
                                    np.array(estimations[i].TOA)
        sorting_indices = np.argsort(cur_toa)
        aoa_values.append(cur_aoa[sorting_indices])
        zoa_values.append(cur_zoa[sorting_indices])
        toa_values.append(cur_toa[sorting_indices])
    return aoa_values, zoa_values, toa_values


def optimize_to_estimate_position_3d(bs_locs, estimations):
    def cost_func_3d(x0):
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
        # scatters constraints
        for i in range(len(bs_locs)):
            # NLOS AOA constraints
            for l, nlos_aoa in enumerate(aoa_values[i][1:]):
                if l >= MAX_L:
                    break
                cost = abs(
                    np.arctan2(scatterers[l][1] - bs_locs[i][1], scatterers[l][0] - bs_locs[i][0]) - nlos_aoa)
                costs.append(cost)
            # NLOS ZOA constraints
            for l, nlos_zoa in enumerate(zoa_values[i][1:]):
                if l >= MAX_L:
                    break
                dist_2d = np.linalg.norm(scatterers[l - 1, :2] - bs_locs[i][:2])
                cost = abs(np.arctan2(dist_2d, bs_locs[i][2] - scatterers[l - 1, 2]) - nlos_zoa)
                costs.append(cost)
            # NLOS TOA constraints
            for l, nlos_toa in enumerate(toa_values[i][1:]):
                if l >= MAX_L:
                    break
                cost = abs((np.linalg.norm(bs_locs[i] - scatterers[l]) +
                            np.linalg.norm(conf.ue_pos - scatterers[l])) / C - nlos_toa)
                costs.append(cost)
        return costs

    aoa_values, zoa_values, toa_values = extract_measurements_from_estimations(bs_locs, estimations)
    # LOS computation of location in case of angle and time estimations, or more than one BS/LOS path
    bs_locs = np.array(bs_locs)
    initial_ue_loc = np.array([[0, 0, 0]])
    initial_scatterers = np.zeros([MAX_L, 3])
    x0 = np.concatenate([initial_ue_loc, initial_scatterers]).reshape(-1)
    res = least_squares(cost_func_3d, x0).x
    est_ue_pos, est_scatterers = res[:3], res[3:].reshape(-1, 3)
    return est_ue_pos
