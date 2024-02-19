import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from dir_definitions import NY_DIR
from python_code import conf
from python_code.main import main

plt.style.use('dark_background')

if __name__ == "__main__":
    ue_pos = np.array([45, 5])
    conf.ue_pos[0] = ue_pos[0]
    conf.ue_pos[1] = ue_pos[1]
    input_powers = np.arange(-50, 101)
    # go over multiple SNRs
    rmse_dict = {}
    for input_power in input_powers:
        conf.input_power = input_power
        rmse = main()
        rmse_dict[input_power] = [rmse, rmse > 1]
    rmse_df = pd.DataFrame.from_dict(rmse_dict, orient='index', columns=['RMSE', 'Error > 1m'])
    if not os.path.exists(f"{NY_DIR}/{ue_pos}"):
        os.makedirs(f"{NY_DIR}/{ue_pos}", exist_ok=True)
    path = f"{NY_DIR}/{ue_pos}/fc_{conf.fc}_antennas_{conf.Nr_x}_bw_{conf.BW}_subcarriers_{conf.K}_band_type_{conf.band_type}"
    rmse_df.to_csv(f"{path}.csv")
    fig = plt.figure()
    plt.plot(input_powers, [rmse_dict[power] for power in input_powers])
    ax = plt.gca()
    ax.set_xlabel('Power[dBm]')
    ax.set_ylabel('RMSE')
    plt.ylim([0, 50])
    plt.savefig(f'{path}.png', dpi=fig.dpi)
    plt.show()
