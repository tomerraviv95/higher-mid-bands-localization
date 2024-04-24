import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from dir_definitions import ROOT_DIR
from python_code import conf
from python_code.main import main

plt.style.use('dark_background')

if __name__ == "__main__":
    rmse_list = []
    ue_x_positions = range(8, 16)
    ue_y_positions = range(8, 16)
    for ue_pos_x in ue_x_positions:
        rmse_list.append([])
        for ue_pos_y in ue_y_positions:
            print('******' * 5)
            conf.ue_pos[0] = ue_pos_x
            conf.ue_pos[1] = ue_pos_y
            rmse = main()
            rmse_list[-1].append(rmse)
    rmse_array = np.flipud(np.round(np.array(rmse_list), 2).T)
    fig = plt.figure(figsize=(12, 8), dpi=80)
    sns.heatmap(rmse_array, annot=True, linewidths=.5, vmin=0, vmax=1)
    ax = plt.gca()
    plt.xlabel('X location')
    plt.xticks(range(len(ue_x_positions)), ue_x_positions)
    plt.ylabel('Y location')
    plt.yticks(range(1, len(ue_y_positions) + 1), ue_y_positions[::-1])
    name = f"{ROOT_DIR}/rmse_synthetic_fc_{conf.fc}_antennas_{conf.Nr}_bw_{conf.BW}_subcarriers_{conf.K}.png"
    plt.savefig(name, dpi=fig.dpi)
