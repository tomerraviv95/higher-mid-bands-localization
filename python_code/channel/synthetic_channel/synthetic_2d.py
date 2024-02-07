# Generate scatter points
from typing import List, Tuple

import numpy as np

from python_code import conf
from python_code.utils.bands_manipulation import Band
from python_code.utils.constants import C, P_0, L_MAX
from python_code.utils.path_loss import compute_path_loss, calc_power


def generate_synthetic_parameters(bs_loc: np.ndarray, ue_pos: np.ndarray, scatterers: np.ndarray, band: Band) -> Tuple[
    List[float], List[float], List[float]]:
    """"
    Computes the parameters for each path. Each path includes the toa, aoa and power.
    """
    # Initialize the channel parameters_2d for L paths
    TOA = [0 for _ in range(L_MAX)]
    AOA = [0 for _ in range(L_MAX)]
    POWER = [0 for _ in range(L_MAX)]
    # Compute for the LOS path
    conf.medium_speed, conf.orientation = C, 0
    TOA[0] = np.linalg.norm(ue_pos - bs_loc) / C
    AOA[0] = np.arctan2(ue_pos[1] - bs_loc[1], ue_pos[0] - bs_loc[0])
    initial_power = P_0 * np.sqrt(1 / 2) * (1 + 1j)
    POWER[0] = calc_power(initial_power, bs_loc, ue_pos, band.fc) / compute_path_loss(TOA[0], band.fc)
    # Compute for the NLOS paths
    for l in range(1, L_MAX):
        AOA[l] = np.arctan2(scatterers[l - 1, 1] - bs_loc[1], scatterers[l - 1, 0] - bs_loc[0])
        TOA[l] = (np.linalg.norm(bs_loc - scatterers[l - 1]) + np.linalg.norm(conf.ue_pos - scatterers[l - 1])) / C
        initial_power = P_0 * np.sqrt(1 / 2) * (1 + 1j)
        POWER[l] = calc_power(calc_power(initial_power, bs_loc, scatterers[l - 1], band.fc), scatterers[l - 1],
                              ue_pos, band.fc) / compute_path_loss(TOA[l], band.fc)
    # assert that the toas are supported, must be smaller than largest distance divided by the speed
    assert all([TOA[l] < band.K / band.BW for l in range(len(TOA))])
    return TOA, AOA, POWER
