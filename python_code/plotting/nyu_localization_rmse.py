import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from dir_definitions import RAYTRACING_DIR, ROOT_DIR
from python_code import conf
from python_code.main import main

plt.style.use('dark_background')

if __name__ == "__main__":
    rmse_dict = {}
    ue_x_positions = range(0, 1121, 5)
    ue_y_positions = range(0, 506, 5)
    csv_path = os.path.join(RAYTRACING_DIR, str(6000), f"bs{str(1)}.csv")
    csv_loaded = pd.read_csv(csv_path)
    count = 0
    MAX_COUNT = 100
    for ue_pos_x in ue_x_positions:
        for ue_pos_y in ue_y_positions:
            if count >= MAX_COUNT:
                break
            ue_pos = np.array([ue_pos_x, ue_pos_y])
            row_ind = csv_loaded.index[(csv_loaded[['rx_x', 'rx_y']] == ue_pos).all(axis=1)].item()
            row = csv_loaded.iloc[row_ind]
            if row['link state'] != 1:
                continue
            print('******' * 5)
            conf.ue_pos[0] = ue_pos_x
            conf.ue_pos[1] = ue_pos_y
            rmse = main()
            rmse_dict[(ue_pos_x, ue_pos_y)] = rmse
            count += 1

    rmse_df = pd.DataFrame.from_dict(rmse_dict, orient='index', columns=['RMSE', 'Error > 1m'])
    rmse_df.loc['mean'] = rmse_df.mean()
    rmse_df.to_csv(f"{ROOT_DIR}/rmse_ny_{conf.fc}_{conf.Nr_x}_{conf.BW}.csv")
