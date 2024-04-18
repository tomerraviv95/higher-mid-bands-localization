import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from dir_definitions import RAYTRACING_DIR, NY_DIR
from python_code import conf
from python_code.main import main

plt.style.use('dark_background')

if __name__ == "__main__":
    # calculation of the rmse results for the figures in the paper
    # compute the average rmse over the 400 locations for each transmitted power
    csv_path = os.path.join(RAYTRACING_DIR, str(6000), f"bs{str(1)}.csv")
    csv_loaded = pd.read_csv(csv_path)
    params6 = {'K': [100], 'Nr': [4], 'fc': [6000], 'BW': [6], 'alg': 'Beamformer'}
    params12 = {'K': [200], 'Nr': [8], 'fc': [12000], 'BW': [12], 'alg': 'Beamformer'}
    params18 = {'K': [150], 'Nr': [16], 'fc': [18000], 'BW': [24], 'alg': 'Beamformer'}
    params24 = {'K': [200], 'Nr': [24], 'fc': [24000], 'BW': [48], 'alg': 'Beamformer'}
    params_6_24 = {'K': [100, 200], 'Nr': [4, 24], 'fc': [6000, 24000], 'BW': [6, 48], 'alg': 'Beamformer'}
    params_all = {'K': [100, 200, 150, 200], 'Nr': [5, 10, 15, 20], 'fc': [6000, 12000, 18000, 24000],
                  'BW': [3, 6, 9, 12], 'alg': 'Beamformer'}
    params_list = [params_all, params6, params12, params18, params24, params_6_24]
    params_list = [params6, params24, params_6_24]
    for params in params_list:
        for field, value in params.items():
            conf.set_value(field=field, value=value)
        ue_x_positions = range(0, 1121, 5)
        ue_y_positions = range(0, 506, 5)
        input_powers = range(-10, 21, 10)
        # go over multiple SNRs
        for input_power in input_powers:
            rmse_dict = {}
            conf.input_power = input_power
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
                    rmse, gt, est = main()
                    rmse_dict[(ue_pos_x, ue_pos_y)] = [rmse, gt[0], gt[1], est[0], est[1]]
            rmse_df = pd.DataFrame.from_dict(rmse_dict, orient='index',
                                             columns=['Position RMSE', 'AOA', 'TOA', 'EST AOA', 'EST TOA'])
            rmse_df.loc['mean'] = rmse_df.mean()
            file_name = f"{conf.alg}_fc_{conf.fc}_antennas_{conf.Nr}_bw_{conf.BW}_subcarriers_{conf.K}.csv"
            if not os.path.exists(f"{NY_DIR}/{str(input_power)}"):
                os.makedirs(f"{NY_DIR}/{str(input_power)}", exist_ok=True)
            rmse_df.to_csv(f"{NY_DIR}/{str(input_power)}/{file_name}")
