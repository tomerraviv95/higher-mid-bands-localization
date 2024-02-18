import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from dir_definitions import RAYTRACING_DIR, NY_DIR
from python_code import conf
from python_code.main import main

plt.style.use('dark_background')

if __name__ == "__main__":
    params1 = {'K':[20],'Nr_x':[6],'Nr_y':[1],'fc':[6000],'BW':[5],'band_type': 'SINGLE'}
    params2 = {'K':[20],'Nr_x':[18],'Nr_y':[1],'fc':[24000],'BW':[10],'band_type': 'SINGLE'}
    params3 = {'K':[ 20 ,20],'Nr_x':[ 6 ,18],'Nr_y':[ 1 ,1],'fc':[ 6000 ,24000],
               'BW':[ 5 ,10],'band_type': 'MULTI'}
    params_list = [params3,params2,params1]
    for params in params_list:
        for field,value in params.items():
            conf.set_value(field=field,value=value)
        ue_x_positions = range(0, 1121, 5)
        ue_y_positions = range(0, 506, 5)
        csv_path = os.path.join(RAYTRACING_DIR, str(6000), f"bs{str(1)}.csv")
        csv_loaded = pd.read_csv(csv_path)
        input_powers = range(-15,21)
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
                    rmse = main()
                    rmse_dict[(ue_pos_x, ue_pos_y)] = [rmse, rmse > 1]
            rmse_df = pd.DataFrame.from_dict(rmse_dict, orient='index', columns=['RMSE', 'Error > 1m'])
            rmse_df.loc['mean'] = rmse_df.mean()
            path = f"{NY_DIR}/{str(input_power)}/fc_{conf.fc}_antennas_{conf.Nr_x}_bw_{conf.BW}_subcarriers_{conf.K}.csv"
            if not os.path.exists(f"{NY_DIR}/{str(input_power)}"):
                os.makedirs(f"{NY_DIR}/{str(input_power)}", exist_ok=True)
            rmse_df.to_csv(
                f"{NY_DIR}/{str(input_power)}/fc_{conf.fc}_antennas_{conf.Nr_x}_bw_{conf.BW}_subcarriers_{conf.K}_band_type_{conf.band_type}.csv")
