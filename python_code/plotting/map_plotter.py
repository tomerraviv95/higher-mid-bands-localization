import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dir_definitions import NY_DIR

if __name__ == "__main__":
    files = [
        "Beamformer_fc_[6000]_antennas_[8]_bw_[2.4]_subcarriers_[20].csv",
        "Beamformer_fc_[24000]_antennas_[32]_bw_[9.6]_subcarriers_[20].csv",
        "Beamformer_fc_[6000, 24000]_antennas_[8, 32]_bw_[2.4, 9.6]_subcarriers_[20, 20].csv",
    ]
    input_power = 5
    dir_path = f"{NY_DIR}/{str(input_power)}/"
    fig, axes = plt.subplots(1, 3)
    cmap = mpl.colors.ListedColormap(['black', 'lightblue', 'red'])
    bounds = [0, 10]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
    colormap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    for ax, file in zip(axes, files):
        file_path = dir_path + file
        df = pd.read_csv(file_path, index_col=0)
        hist = -1 * np.ones([200, 200])
        for loc, rmse in zip(df.index[:-1], df['Position RMSE'][:-1]):
            loc = eval(loc)
            loc = (loc[0] // 5, loc[1] // 5)
            hist[loc] = rmse
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 100])
        ax.set_xticks(range(0, 101, 20), ['0', '20', '40', '60', '80', '100'])
        ax.set_yticks(range(0, 101, 20), ['0', '20', '40', '60', '80', '100'])
        ax.set_xlabel('X-Axis [m]')
        ax.set_ylabel('Y-Axis [m]')
        ax.imshow(hist, interpolation='none', cmap=cmap, norm=norm)
    plt.colorbar(colormap, label="RMSE Colormap")
    plt.show()
