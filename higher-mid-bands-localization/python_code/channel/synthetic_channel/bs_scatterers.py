import numpy as np


def create_scatter_points(L: int) -> np.ndarray:
    scatterers = np.array([[15, 12], [15, 14], [12, 14]])
    return scatterers[:L - 1]


def create_bs_locs(bs_ind: int) -> np.ndarray:
    bs_locs = np.array([[0, 0], [0, -5], [0, 5]])
    return bs_locs[bs_ind - 1]
