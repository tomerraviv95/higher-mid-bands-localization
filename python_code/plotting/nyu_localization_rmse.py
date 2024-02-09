import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from dir_definitions import RAYTRACING_DIR, NY_DIR
from python_code import conf
from python_code.main import main

plt.style.use('dark_background')

if __name__ == "__main__":
    ue_x_positions = range(0, 1121, 5)
    ue_y_positions = range(0, 506, 5)
    csv_path = os.path.join(RAYTRACING_DIR, str(6000), f"bs{str(1)}.csv")
    csv_loaded = pd.read_csv(csv_path)
    snrs = [40, 50, 60]
    # go over multiple SNRs
    for snr in snrs:
        rmse_dict = {}
        conf.snr = snr
        # for multiple locations of the UE
        for ue_pos_x in ue_x_positions:
            for ue_pos_y in ue_y_positions:
                ue_pos = np.array([ue_pos_x, ue_pos_y])
                row_ind = csv_loaded.index[(csv_loaded[['rx_x', 'rx_y']] == ue_pos).all(axis=1)].item()
                row = csv_loaded.iloc[row_ind]
                # only compute if the ue in LOS conditions
                if row['link state'] != 1:
                    continue
                print('******' * 5)
                conf.ue_pos[0] = ue_pos_x
                conf.ue_pos[1] = ue_pos_y
                rmse = main()
                rmse_dict[(ue_pos_x, ue_pos_y)] = [rmse, rmse > 1]
        rmse_df = pd.DataFrame.from_dict(rmse_dict, orient='index', columns=['RMSE', 'Error > 1m'])
        rmse_df.loc['mean'] = rmse_df.mean()
        path = f"{NY_DIR}/{str(snr)}/fc_{conf.fc}_antennas_{conf.Nr_x}_bw_{conf.BW}_subcarriers_{conf.K}.csv"
        if not os.path.exists(f"{NY_DIR}/{str(snr)}"):
            os.makedirs(f"{NY_DIR}/{str(snr)}", exist_ok=True)
        rmse_df.to_csv(f"{NY_DIR}/{str(snr)}/fc_{conf.fc}_antennas_{conf.Nr_x}_bw_{conf.BW}_subcarriers_{conf.K}.csv")
