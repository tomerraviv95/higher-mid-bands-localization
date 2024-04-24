from typing import List, Tuple

import numpy as np

from python_code import conf
from python_code.utils.bands_manipulation import Band
from python_code.utils.constants import C, SYNTHETIC_L_MAX
from python_code.utils.path_loss import compute_path_loss, calc_power


def generate_synthetic_parameters(bs_loc: np.ndarray, ue_pos: np.ndarray, scatterers: np.ndarray, band: Band) -> Tuple[
    List[float], List[float], List[float]]:
    """"
    Generate the parameters for each of the L_MAX paths in the current bs-ue link.
    Each path includes the toa, aoa and power.
    """
    # Initialize the channel parameters for L paths
    toas = [0 for _ in range(SYNTHETIC_L_MAX)]
    aoas = [0 for _ in range(SYNTHETIC_L_MAX)]
    powers = [0 for _ in range(SYNTHETIC_L_MAX)]
    # Compute for the LOS path
    conf.medium_speed = C
    toas[0] = np.linalg.norm(ue_pos - bs_loc) / C
    aoas[0] = np.arctan2(ue_pos[1] - bs_loc[1], ue_pos[0] - bs_loc[0])
    initial_power = np.sqrt(1 / 2) * (1 + 1j)
    powers[0] = calc_power(initial_power, bs_loc, ue_pos, band.fc) / compute_path_loss(toas[0], band.fc)
    # Compute for the NLOS paths
    for l in range(1, SYNTHETIC_L_MAX):
        aoas[l] = np.arctan2(scatterers[l - 1, 1] - bs_loc[1], scatterers[l - 1, 0] - bs_loc[0])
        toas[l] = (np.linalg.norm(bs_loc - scatterers[l - 1]) + np.linalg.norm(conf.ue_pos - scatterers[l - 1])) / C
        initial_power = np.sqrt(1 / 2) * (1 + 1j)
        powers[l] = calc_power(calc_power(initial_power, bs_loc, scatterers[l - 1], band.fc), scatterers[l - 1],
                              ue_pos, band.fc) / compute_path_loss(toas[l], band.fc)
    # assert that the toas are supported, must be smaller than largest distance divided by the speed
    assert all([toas[l] < band.K / band.BW for l in range(len(toas))])
    return toas, aoas, powers
