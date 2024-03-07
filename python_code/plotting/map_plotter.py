import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from dir_definitions import NY_DIR

if __name__ == "__main__":
    files = [
        "Beamformer_fc_[6000]_antennas_[8]_bw_[2.4]_subcarriers_[20].csv",
        "Beamformer_fc_[24000]_antennas_[32]_bw_[9.6]_subcarriers_[20].csv",
        "Beamformer_fc_[6000, 24000]_antennas_[8, 32]_bw_[2.4, 9.6]_subcarriers_[20, 20].csv",
    ]
    input_power = -5
    dir_path = f"{NY_DIR}/{str(input_power)}/"
    fig, axes = plt.subplots(1, 3, figsize=(14, 7), sharey=True, layout='constrained')
    # Create colormap for in-range values
    colors = [(0, 0, 0), (0, 0, 1), (1, 0, 0)]  # first color is black, last is red
    cmap_in_range = LinearSegmentedColormap.from_list("Custom", colors, N=7)
    norm_in_range = mpl.colors.Normalize(vmin=-25, vmax=25)
    colormap_in_range = mpl.cm.ScalarMappable(norm=norm_in_range, cmap=cmap_in_range)
    # Create colormap for label
    cmap_for_label = LinearSegmentedColormap.from_list("Custom", colors[1:], N=4)
    norm_for_label = mpl.colors.Normalize(vmin=0, vmax=25)
    colormap_for_label = mpl.cm.ScalarMappable(norm=norm_for_label, cmap=cmap_for_label)
    # colormap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    for ax, file in zip(axes, files):
        file_path = dir_path + file
        df = pd.read_csv(file_path, index_col=0)
        hist = -25 * np.ones([200, 160])
        for loc, rmse in zip(df.index[:-1], df['Position RMSE'][:-1]):
            loc = eval(loc)
            loc = (loc[0] // 5, loc[1] // 5)
            hist[loc] = rmse
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 80])
        ax.set_xticks(range(0, 101, 20), ['0', '20', '40', '60', '80', '100'])
        ax.set_yticks(range(0, 81, 20), ['0', '20', '40', '60', '80'])
        ax.set_xlabel('X-Axis [m]')
        ax.set_ylabel('Y-Axis [m]')
        ax.imshow(hist, interpolation='none', cmap=cmap_in_range, norm=norm_in_range)
    cbar = fig.colorbar(colormap_for_label, fraction=0.042, pad=0.04)
    cbar.outline.set_linewidth(1)
    cbar.ax.set_title(label="RMSE", rotation=0)
    plt.show()
