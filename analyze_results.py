import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dir_definitions import NY_DIR

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['font.size'] = 8
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.figsize'] = [9.5, 6.45]
mpl.rcParams['axes.titlesize'] = 28
mpl.rcParams['axes.labelsize'] = 28
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

file_to_label = {0: "6GHz Sub-band",
                 1: "12GHz Sub-band",
                 2: "18GHz Sub-band",
                 3: "24GHz Sub-band",
                 4: "Proposed - {6,24}GHz",
                 5: "Proposed - {6,12,18,24}GHz"}

color_to_label = {0: "blue",
                  1: "red",
                  2: "pink",
                  3: "green",
                  4: "black",
                  5:"orange"}

marker_to_label = {0: "o",
                   1: "X",
                   2: "s",
                   3: "p",
                   4: "^",
                   5: "P"}

if __name__ == "__main__":
    input_powers = range(0, 251, 10)
    files = ["fc_[6000]_antennas_[5]_bw_[5]_subcarriers_[40]_band_type_SINGLE.csv",
             "fc_[12000]_antennas_[10]_bw_[10]_subcarriers_[40]_band_type_SINGLE.csv",
             "fc_[18000]_antennas_[15]_bw_[15]_subcarriers_[40]_band_type_SINGLE.csv",
             "fc_[24000]_antennas_[20]_bw_[20]_subcarriers_[40]_band_type_SINGLE.csv",
             "fc_[6000, 24000]_antennas_[5, 20]_bw_[5, 20]_subcarriers_[40, 40]_band_type_MULTI.csv"]
    mean_rmse_dict = {}
    mean_errors_dict = {}
    for input_power in input_powers:
        dir_path = f"{NY_DIR}/{str(input_power)}/"
        for file in files:
            file_path = dir_path + file
            df = pd.read_csv(file_path, index_col=0)
            mean_rmse = np.clip(float(df[df.index == 'mean']['RMSE'].item()),a_min=0,a_max=50)
            mean_error = float(df[df.index == 'mean']['Error > 1m'].item())
            if file not in mean_rmse_dict:
                mean_rmse_dict[file] = []
                mean_errors_dict[file] = []
            mean_rmse_dict[file].append(mean_rmse)
            mean_errors_dict[file].append(mean_error)
    # RMSE PLOT
    fig = plt.figure()
    for i, file in enumerate(files):
        plt.plot(input_powers, mean_rmse_dict[file], label=file_to_label[i], markersize=9,
                 linewidth=3.5, color=color_to_label[i], marker=marker_to_label[i])
    plt.xlabel('Transmitted power [dBm]')
    plt.ylabel('RMSE')
    plt.grid(which='both', ls='--')
    plt.legend(loc='upper left', prop={'size': 15})
    plt.ylim([0, 50])
    fig.savefig('RMSE_all.png')
    plt.show()
