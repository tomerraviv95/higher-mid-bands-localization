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

file_to_label = {"fc_[6000]_antennas_[5]_bw_[5]_subcarriers_[40]_band_type_SINGLE.csv": "6GHz Sub-band Capon",
                 "fc_[12000]_antennas_[10]_bw_[10]_subcarriers_[40]_band_type_SINGLE.csv": "12GHz Sub-band Capon",
                 "fc_[18000]_antennas_[15]_bw_[15]_subcarriers_[40]_band_type_SINGLE.csv": "18GHz Sub-band Capon",
                 "fc_[24000]_antennas_[20]_bw_[20]_subcarriers_[40]_band_type_SINGLE.csv": "24GHz Sub-band Capon",
                 "fc_[6000, 24000]_antennas_[5, 20]_bw_[5, 20]_subcarriers_[40, 40]_band_type_MULTI.csv": "{6,24}GHz Multi-Frequency Capon",
                 "fc_[6000, 12000, 18000, 24000]_antennas_[5, 10, 15, 20]_bw_[5, 10, 15, 20]_subcarriers_[40, 40, 40, 40]_band_type_MULTI.csv": "{6,12,18,24}GHz Multi-Frequency Capon"}

color_to_label = {"fc_[6000]_antennas_[5]_bw_[5]_subcarriers_[40]_band_type_SINGLE.csv": "blue",
                  "fc_[12000]_antennas_[10]_bw_[10]_subcarriers_[40]_band_type_SINGLE.csv": "green",
                  "fc_[18000]_antennas_[15]_bw_[15]_subcarriers_[40]_band_type_SINGLE.csv": "orange",
                  "fc_[24000]_antennas_[20]_bw_[20]_subcarriers_[40]_band_type_SINGLE.csv": "red",
                  "fc_[6000, 24000]_antennas_[5, 20]_bw_[5, 20]_subcarriers_[40, 40]_band_type_MULTI.csv": "purple",
                  "fc_[6000, 12000, 18000, 24000]_antennas_[5, 10, 15, 20]_bw_[5, 10, 15, 20]_subcarriers_[40, 40, 40, 40]_band_type_MULTI.csv": "black"}

marker_to_label = {"fc_[6000]_antennas_[5]_bw_[5]_subcarriers_[40]_band_type_SINGLE.csv": "o",
                   "fc_[12000]_antennas_[10]_bw_[10]_subcarriers_[40]_band_type_SINGLE.csv": "X",
                   "fc_[18000]_antennas_[15]_bw_[15]_subcarriers_[40]_band_type_SINGLE.csv": "s",
                   "fc_[24000]_antennas_[20]_bw_[20]_subcarriers_[40]_band_type_SINGLE.csv": "p",
                   "fc_[6000, 24000]_antennas_[5, 20]_bw_[5, 20]_subcarriers_[40, 40]_band_type_MULTI.csv": "^",
                   "fc_[6000, 12000, 18000, 24000]_antennas_[5, 10, 15, 20]_bw_[5, 10, 15, 20]_subcarriers_[40, 40, 40, 40]_band_type_MULTI.csv": "P"}

linestyle_to_label = {"fc_[6000]_antennas_[5]_bw_[5]_subcarriers_[40]_band_type_SINGLE.csv": "dotted",
                   "fc_[12000]_antennas_[10]_bw_[10]_subcarriers_[40]_band_type_SINGLE.csv": "dashed",
                   "fc_[18000]_antennas_[15]_bw_[15]_subcarriers_[40]_band_type_SINGLE.csv": "dashdot",
                   "fc_[24000]_antennas_[20]_bw_[20]_subcarriers_[40]_band_type_SINGLE.csv": (5,(10,3)),
                   "fc_[6000, 24000]_antennas_[5, 20]_bw_[5, 20]_subcarriers_[40, 40]_band_type_MULTI.csv": "solid",
                   "fc_[6000, 12000, 18000, 24000]_antennas_[5, 10, 15, 20]_bw_[5, 10, 15, 20]_subcarriers_[40, 40, 40, 40]_band_type_MULTI.csv": "solid"}


if __name__ == "__main__":
    input_powers = range(0, 101, 5)
    files1 = [
        "fc_[6000]_antennas_[5]_bw_[5]_subcarriers_[40]_band_type_SINGLE.csv",
        "fc_[12000]_antennas_[10]_bw_[10]_subcarriers_[40]_band_type_SINGLE.csv",
        "fc_[18000]_antennas_[15]_bw_[15]_subcarriers_[40]_band_type_SINGLE.csv",
        "fc_[24000]_antennas_[20]_bw_[20]_subcarriers_[40]_band_type_SINGLE.csv",
        "fc_[6000, 12000, 18000, 24000]_antennas_[5, 10, 15, 20]_bw_[5, 10, 15, 20]_subcarriers_[40, 40, 40, 40]_band_type_MULTI.csv"
        ]
    files2 = [
        "fc_[6000]_antennas_[5]_bw_[5]_subcarriers_[40]_band_type_SINGLE.csv",
        "fc_[24000]_antennas_[20]_bw_[20]_subcarriers_[40]_band_type_SINGLE.csv",
        "fc_[6000, 24000]_antennas_[5, 20]_bw_[5, 20]_subcarriers_[40, 40]_band_type_MULTI.csv",
    ]
    files = files1
    mean_rmse_dict = {}
    for input_power in input_powers:
        dir_path = f"{NY_DIR}/{str(input_power)}/"
        for file in files:
            file_path = dir_path + file
            df = pd.read_csv(file_path, index_col=0)
            mean_rmse = df[df.index == 'mean']['Position RMSE'].item()
            if file not in mean_rmse_dict:
                mean_rmse_dict[file] = []
            mean_rmse_dict[file].append(mean_rmse)
    # RMSE PLOT
    fig = plt.figure()
    for file in files:
        plt.plot(input_powers, mean_rmse_dict[file], label=file_to_label[file], markersize=9,
                 linewidth=3.5, color=color_to_label[file], marker=marker_to_label[file],
                 linestyle = linestyle_to_label[file])
    plt.xlabel('Transmitted Power [dBm]')
    plt.ylabel('RMSE')
    plt.grid(which='both', ls='--')
    plt.legend(loc='upper left', prop={'size': 15})
    plt.ylim([0, 30])
    fig.savefig('RMSE.png')
    plt.show()
