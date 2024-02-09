import math
import os

import numpy as np
import pandas as pd

from dir_definitions import RAYTRACING_DIR
from python_code import conf
from python_code.utils.bands_manipulation import Band
from python_code.utils.constants import MU_SEC
from python_code.utils.path_loss import watt_from_db


def load_ny_scenario(bs_ind: int, ue_pos: np.ndarray, band: Band):
    csv_path = os.path.join(RAYTRACING_DIR, str(band.fc), f"bs{str(bs_ind)}.csv")
    csv_loaded = pd.read_csv(csv_path)
    row_ind = csv_loaded.index[(csv_loaded[['rx_x', 'rx_y']] == ue_pos).all(axis=1)].item()
    row = csv_loaded.iloc[row_ind]
    if row['link state'] != 1:
        raise ValueError("NLOS location! currently supporting only LOS")
    bs_loc = np.array(row[['tx_x', 'tx_y']]).astype(float)
    n_paths = row['n_path'].astype(int)
    powers, toas, aoas = [], [], []
    for path in range(1, n_paths + 1):
        initial_power = 0.5 * (1 + 1j)
        loss_db = row[f'path_loss_{path}']
        received_power = initial_power * (1 / watt_from_db(loss_db))
        toa = row[f'delay_{path}'] / MU_SEC
        if path == 1:
            conf.medium_speed = np.linalg.norm(ue_pos - bs_loc) / toa
        # path is above the maximal range, so ignore it
        if toa > band.K / band.BW:
            continue
        aoa = math.radians(row[f'aod_{path}'])
        if path == 1:
            conf.orientation = aoa
        normalized_aoa = aoa - conf.orientation
        # the base station can see 90 degrees to each side of its orientation
        if -math.pi / 2 < normalized_aoa < math.pi / 2:
            powers.append(received_power), toas.append(toa), aoas.append(normalized_aoa)
    assert all([toas[l] < band.K / band.BW for l in range(len(toas))])
    return bs_loc, toas, aoas, powers
