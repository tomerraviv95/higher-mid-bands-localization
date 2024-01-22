import numpy as np
from matplotlib import pyplot as plt

from python_code import conf
from python_code.main import main

plt.style.use('dark_background')

if __name__ == "__main__":
    rmse_list = [[]]
    for bs_count in range(1, 4):
        conf.B = bs_count
        for l in range(1, 4):
            conf.L = l
            rmse = main()
            rmse_list[-1].append(rmse)
        rmse_list.append([])
    del rmse_list[-1]
    rmse_array = np.round(np.array(rmse_list), 2)
    plt.imshow(rmse_array, interpolation='nearest')
    ax = plt.gca()
    plt.xticks(range(3), range(1, 4))
    plt.ylabel('BS Count')
    plt.xlabel('LOS+NLOS Paths')
    plt.yticks(range(3), range(1, 4))
    for i in range(3):
        for j in range(3):
            text = ax.text(j, i, rmse_array[i, j],
                           ha="center", va="center", color="black")
    plt.show()
