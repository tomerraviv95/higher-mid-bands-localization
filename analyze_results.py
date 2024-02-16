import matplotlib as mpl
import matplotlib.pyplot as plt
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

file_to_label = {"fc_[6000]_antennas_[6]_bw_[5]_subcarriers_[80]_band_type_SINGLE.csv": "6GHz Sub-band",
                 "fc_[24000]_antennas_[24]_bw_[20]_subcarriers_[80]_band_type_SINGLE.csv": "24GHz Sub-band",
                 "fc_[6000, 24000]_antennas_[6, 24]_bw_[5, 20]_subcarriers_[80, 80]_band_type_SINGLE.csv": "Naive - {6,24}GHz",
                 "fc_[6000, 24000]_antennas_[6, 24]_bw_[5, 20]_subcarriers_[80, 80]_band_type_MULTI.csv": "Proposed - {6,24}GHz"}

color_to_label = {"fc_[6000]_antennas_[6]_bw_[5]_subcarriers_[80]_band_type_SINGLE.csv": "blue",
                  "fc_[24000]_antennas_[24]_bw_[20]_subcarriers_[80]_band_type_SINGLE.csv": "red",
                  "fc_[6000, 24000]_antennas_[6, 24]_bw_[5, 20]_subcarriers_[80, 80]_band_type_SINGLE.csv": "black",
                  "fc_[6000, 24000]_antennas_[6, 24]_bw_[5, 20]_subcarriers_[80, 80]_band_type_MULTI.csv": "purple"}

marker_to_label = {"fc_[6000]_antennas_[6]_bw_[5]_subcarriers_[80]_band_type_SINGLE.csv": "o",
                   "fc_[24000]_antennas_[24]_bw_[20]_subcarriers_[80]_band_type_SINGLE.csv": "X",
                   "fc_[6000, 24000]_antennas_[6, 24]_bw_[5, 20]_subcarriers_[80, 80]_band_type_SINGLE.csv": "^",
                   "fc_[6000, 24000]_antennas_[6, 24]_bw_[5, 20]_subcarriers_[80, 80]_band_type_MULTI.csv": "s"}

if __name__ == "__main__":
    input_powers = [-15, -10, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 10]
    files = ["fc_[6000]_antennas_[6]_bw_[5]_subcarriers_[80]_band_type_SINGLE.csv",
             "fc_[24000]_antennas_[24]_bw_[20]_subcarriers_[80]_band_type_SINGLE.csv",
             "fc_[6000, 24000]_antennas_[6, 24]_bw_[5, 20]_subcarriers_[80, 80]_band_type_SINGLE.csv",
             "fc_[6000, 24000]_antennas_[6, 24]_bw_[5, 20]_subcarriers_[80, 80]_band_type_MULTI.csv"]
    mean_rmse_dict = {}
    mean_errors_dict = {}
    for input_power in input_powers:
        dir_path = f"{NY_DIR}/{str(input_power)}/"
        for file in files:
            file_path = dir_path + file
            df = pd.read_csv(file_path, index_col=0)
            mean_rmse = float(df[df.index == 'mean']['RMSE'].item())
            mean_error = float(df[df.index == 'mean']['Error > 1m'].item())
            if file not in mean_rmse_dict:
                mean_rmse_dict[file] = []
                mean_errors_dict[file] = []
            mean_rmse_dict[file].append(mean_rmse)
            mean_errors_dict[file].append(mean_error)
    ### RMSE PLOT ###
    plt.figure()
    for file in files:
        plt.plot(input_powers, mean_rmse_dict[file], label=file_to_label[file], markersize=9,
                 linewidth=3.5, color=color_to_label[file], marker=marker_to_label[file])
    plt.xlabel('Transmitted power [dBm]')
    plt.ylabel('RMSE')
    plt.grid(which='both', ls='--')
    plt.legend(loc='upper right', prop={'size': 15})
    plt.ylim([0, 30])
    plt.show()
    ### Error Rate PLOT ###
    plt.figure()
    for file in files:
        plt.plot(input_powers, mean_errors_dict[file], label=file_to_label[file], markersize=9,
                 linewidth=3.5, color=color_to_label[file], marker=marker_to_label[file])
    plt.xlabel('Transmitted power [dBm]')
    plt.ylabel('Error Rate (RMSE>1m)')
    plt.grid(which='both', ls='--')
    plt.legend(loc='upper right', prop={'size': 15})
    plt.show()
