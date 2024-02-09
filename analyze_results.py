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

file_to_label = {"fc_[6000]_antennas_[8]_bw_[6]_subcarriers_[24].csv": "6GHz Sub-band",
                 "fc_[24000]_antennas_[32]_bw_[12]_subcarriers_[48].csv": "24GHz Sub-band",
                 "fc_[6000, 24000]_antennas_[8, 32]_bw_[6, 12]_subcarriers_[24, 48].csv": "6GHz + 24GHz Sub-bands"}

color_to_label = {"fc_[6000]_antennas_[8]_bw_[6]_subcarriers_[24].csv": "blue",
                  "fc_[24000]_antennas_[32]_bw_[12]_subcarriers_[48].csv": "red",
                  "fc_[6000, 24000]_antennas_[8, 32]_bw_[6, 12]_subcarriers_[24, 48].csv": "purple"}

marker_to_label = {"fc_[6000]_antennas_[8]_bw_[6]_subcarriers_[24].csv": "o",
                   "fc_[24000]_antennas_[32]_bw_[12]_subcarriers_[48].csv": "X",
                   "fc_[6000, 24000]_antennas_[8, 32]_bw_[6, 12]_subcarriers_[24, 48].csv": "s"}

if __name__ == "__main__":
    snrs = [40, 50, 60]
    files = ["fc_[6000]_antennas_[8]_bw_[6]_subcarriers_[24].csv",
             "fc_[24000]_antennas_[32]_bw_[12]_subcarriers_[48].csv",
             "fc_[6000, 24000]_antennas_[8, 32]_bw_[6, 12]_subcarriers_[24, 48].csv"]
    mean_errors_dict = {}
    for snr in snrs:
        dir_path = f"{NY_DIR}/{str(snr)}/"
        for file in files:
            file_path = dir_path + file
            df = pd.read_csv(file_path, index_col=0)
            mean_error = float(df[df.index == 'mean']['Error > 1m'].item())
            if file not in mean_errors_dict:
                mean_errors_dict[file] = []
            mean_errors_dict[file].append(mean_error)
    print(mean_errors_dict)
    plt.figure()
    for file in files:
        plt.plot(snrs, mean_errors_dict[file], label=file_to_label[file], markersize=11, linewidth=2.5,
                 color=color_to_label[file], marker=marker_to_label[file])
    plt.xlabel('SNR[dB]')
    plt.ylabel('RMSE')
    plt.grid(which='both', ls='--')
    plt.legend(loc='upper right', prop={'size': 15})
    plt.show()
