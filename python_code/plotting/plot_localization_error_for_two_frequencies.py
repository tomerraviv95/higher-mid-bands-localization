import numpy as np
from matplotlib import pyplot as plt

from python_code import conf
from python_code.main import main

plt.style.use('dark_background')

if __name__ == "__main__":
    rmse_list = []
    ue_x_positions = range(9, 10)
    ue_y_positions = range(9, 10)
    for ue_pos_x in ue_x_positions:
        rmse_list.append([])
        for ue_pos_y in ue_y_positions:
            conf.ue_pos[0] = ue_pos_x
            conf.ue_pos[1] = ue_pos_y
            rmse = main()
            rmse_list[-1].append(rmse)
    rmse_array = np.round(np.array(rmse_list), 2)
    plt.imshow(rmse_array, interpolation='nearest')
    ax = plt.gca()
    plt.xticks(ue_x_positions, ue_x_positions)
    plt.ylabel('Y lcoation')
    plt.xlabel('X location')
    plt.yticks(ue_y_positions, ue_y_positions)
    for i in ue_x_positions:
        for j in ue_y_positions:
            text = ax.text(j, i, rmse_array[i, j],
                           ha="center", va="center", color="black")
    plt.show()
