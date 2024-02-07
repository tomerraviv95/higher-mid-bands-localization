import os

import numpy as np
import pandas as pd

from dir_definitions import RAYTRACING_DIR
from python_code.utils.bands_manipulation import Band
from python_code.utils.constants import mu_sec, P_0


def load_ny_scenario(bs_ind: int, ue_pos: np.ndarray, band: Band):
    csv_path = os.path.join(RAYTRACING_DIR, str(band.fc), f"bs{str(bs_ind)}.csv")
    csv_loaded = pd.read_csv(csv_path)
    row_ind = csv_loaded.index[(csv_loaded[['rx_x', 'rx_y']] == ue_pos).all(axis=1)].item()
    row = csv_loaded.iloc[row_ind]
    if row['link state'] == 2:
        raise ValueError("NLOS location! currently supporting only LOS")
    bs_loc = np.array(row[['tx_x', 'tx_y']]).astype(float)
    n_paths = row['n_path'].astype(int)
    powers, toas, aoas = [], [], []
    for path in range(1, n_paths + 1):
        initial_power = P_0 * np.sqrt(1 / 2) * (1 + 1j)
        loss_db = row[f'path_loss_{path}']
        power = initial_power / 10 ** (loss_db / 20)
        toa = row[f'delay_{path}'] / mu_sec
        aoa = row[f'aod_{path}']
        powers.append(power), toas.append(toa), aoas.append(aoa)

    return bs_loc, toas, aoas, powers
