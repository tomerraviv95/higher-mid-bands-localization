import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dir_definitions import NY_DIR

MAX_RMSE = 100

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['font.size'] = 6
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.figsize'] = [9.5, 6.45]
mpl.rcParams['axes.titlesize'] = 28
mpl.rcParams['axes.labelsize'] = 28
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

file_to_label = {"Beamformer_fc_[6000]_antennas_[4]_bw_[6]_subcarriers_[100].csv": "6GHz Sub-band Beamformer",
                 "Beamformer_fc_[12000]_antennas_[8]_bw_[12]_subcarriers_[200].csv": "12GHz Sub-band Beamformer",
                 "Beamformer_fc_[18000]_antennas_[16]_bw_[24]_subcarriers_[150].csv": "18GHz Sub-band Beamformer",
                 "Beamformer_fc_[24000]_antennas_[24]_bw_[48]_subcarriers_[200].csv": "24GHz Sub-band Beamformer",
                 "Beamformer_fc_[6000, 24000]_antennas_[4, 24]_bw_[6, 48]_subcarriers_[100, 200].csv": "{6,24}GHz Multi-Frequency Beamformer",
                 "Beamformer_fc_[6000, 12000, 18000, 24000]_antennas_[4, 8, 16, 24]_bw_[6, 12, 24, 48]_subcarriers_[100, 200, 150, 200].csv": "{6,12,18,24}GHz Multi-Frequency Beamformer"
                 }

color_to_label = {"Beamformer_fc_[6000]_antennas_[4]_bw_[6]_subcarriers_[100].csv": "blue",
                  "Beamformer_fc_[12000]_antennas_[8]_bw_[12]_subcarriers_[200].csv": "green",
                  "Beamformer_fc_[18000]_antennas_[16]_bw_[24]_subcarriers_[150].csv": "orange",
                  "Beamformer_fc_[24000]_antennas_[24]_bw_[48]_subcarriers_[200].csv": "red",
                  "Beamformer_fc_[6000, 24000]_antennas_[4, 24]_bw_[6, 48]_subcarriers_[100, 200].csv": "black",
                  "Beamformer_fc_[6000, 12000, 18000, 24000]_antennas_[4, 8, 16, 24]_bw_[6, 12, 24, 48]_subcarriers_[100, 200, 150, 200].csv": "black"}

marker_to_label = {"Beamformer_fc_[6000]_antennas_[4]_bw_[6]_subcarriers_[100].csv": "o",
                   "Beamformer_fc_[12000]_antennas_[8]_bw_[12]_subcarriers_[200].csv": "X",
                   "Beamformer_fc_[18000]_antennas_[16]_bw_[24]_subcarriers_[150].csv": "s",
                   "Beamformer_fc_[24000]_antennas_[24]_bw_[48]_subcarriers_[200].csv": "p",
                   "Beamformer_fc_[6000, 24000]_antennas_[4, 24]_bw_[6, 48]_subcarriers_[100, 200].csv": "P",
                   "Beamformer_fc_[6000, 12000, 18000, 24000]_antennas_[4, 8, 16, 24]_bw_[6, 12, 24, 48]_subcarriers_[100, 200, 150, 200].csv": "P"}

linestyle_to_label = {"Beamformer_fc_[6000]_antennas_[4]_bw_[6]_subcarriers_[100].csv": "dotted",
                      "Beamformer_fc_[12000]_antennas_[8]_bw_[12]_subcarriers_[200].csv": "dashed",
                      "Beamformer_fc_[18000]_antennas_[16]_bw_[24]_subcarriers_[150].csv": "dashdot",
                      "Beamformer_fc_[24000]_antennas_[24]_bw_[48]_subcarriers_[200].csv": (5, (10, 3)),
                      "Beamformer_fc_[6000, 24000]_antennas_[4, 24]_bw_[6, 48]_subcarriers_[100, 200].csv": "solid",
                      "Beamformer_fc_[6000, 12000, 18000, 24000]_antennas_[4, 8, 16, 24]_bw_[6, 12, 24, 48]_subcarriers_[100, 200, 150, 200].csv": "solid"}

if __name__ == "__main__":
    # plotter for rmse results figures in the paper
    input_powers = range(-20, 41, 10)
    files1 = [
        "Beamformer_fc_[6000]_antennas_[4]_bw_[6]_subcarriers_[100].csv",
        "Beamformer_fc_[12000]_antennas_[8]_bw_[12]_subcarriers_[200].csv",
        "Beamformer_fc_[18000]_antennas_[16]_bw_[24]_subcarriers_[150].csv",
        "Beamformer_fc_[24000]_antennas_[24]_bw_[48]_subcarriers_[200].csv",
        # "Beamformer_fc_[6000, 24000]_antennas_[4, 24]_bw_[6, 48]_subcarriers_[100, 200].csv",
        "Beamformer_fc_[6000, 12000, 18000, 24000]_antennas_[4, 8, 16, 24]_bw_[6, 12, 24, 48]_subcarriers_[100, 200, 150, 200].csv"
    ]
    files2 = [
        "Beamformer_fc_[6000]_antennas_[4]_bw_[6]_subcarriers_[100].csv",
        "Beamformer_fc_[24000]_antennas_[24]_bw_[48]_subcarriers_[200].csv",
        "Beamformer_fc_[6000, 24000]_antennas_[4, 24]_bw_[6, 48]_subcarriers_[100, 200].csv",
    ]
    files = files1
    mean_rmse_dict = {}
    for input_power in input_powers:
        dir_path = f"{NY_DIR}/{str(input_power)}/"
        for file in files:
            file_path = dir_path + file
            df = pd.read_csv(file_path, index_col=0)
            mean_rmse = np.mean(np.clip(df['Position RMSE'], a_min=0, a_max=MAX_RMSE))
            if file not in mean_rmse_dict:
                mean_rmse_dict[file] = []
            mean_rmse_dict[file].append(mean_rmse)
    # RMSE PLOT
    fig = plt.figure()
    for file in files:
        plt.plot(input_powers, mean_rmse_dict[file], label=file_to_label[file], markersize=9,
                 linewidth=3.5, color=color_to_label[file], marker=marker_to_label[file],
                 linestyle=linestyle_to_label[file])
    plt.xlabel('Transmitted Power [dBm]')
    plt.ylabel('RMSE [m]')
    plt.grid(which='both', ls='--')
    plt.legend(loc='lower left', prop={'size': 13})
    plt.ylim([0, 5])
    fig.savefig('RMSE.png')
    plt.show()
