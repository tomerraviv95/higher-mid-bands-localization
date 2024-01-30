import numpy as np

from python_code import conf
from python_code.utils.constants import DimensionType


def create_scatter_points_2d(L: int):
    scatterers = np.array([[4, 4], [3, -6], [16, -16], [20, 20]])
    return scatterers[:L - 1]


def create_scatter_points_3d(L: int):
    scatterers = np.array([[4, 4, 4], [3, -6, 3], [16, -16, 8], [20, 20, 0]])
    return scatterers[:L - 1]


def create_scatter_points(L: int):
    assert L > 0
    if conf.dimensions == DimensionType.Three.name:
        return create_scatter_points_3d(L)
    else:
        return create_scatter_points_2d(L)


def create_bs_locs_2d(B: int):
    bs_locs = np.array([[0, 5], [0, -5], [0, 50]])
    return bs_locs[:B]


def create_bs_locs_3d(B: int):
    bs_locs = np.array([[0, 5, 10], [0, -5, 10], [0, 50, 10]])
    return bs_locs[:B]


def create_bs_locs(B: int):
    assert B > 0
    if conf.dimensions == DimensionType.Three.name:
        return create_bs_locs_3d(B)
    else:
        return create_bs_locs_2d(B)
