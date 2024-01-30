import numpy as np

from python_code import conf
from python_code.utils.constants import DimensionType


def create_scatter_points_2d(L: int):
    scatterers = np.array([[7, -5], [10, -6], [10, 3]])
    return scatterers[:L - 1]


def create_scatter_points_3d(L: int):
    scatterers = np.array([[10, 2, 4], [3, -16, 3], [5, -5, 8]])
    return scatterers[:L - 1]


def create_scatter_points(L: int):
    assert L > 0
    if conf.dimensions == DimensionType.Three.name:
        return create_scatter_points_3d(L)
    else:
        return create_scatter_points_2d(L)


def create_bs_locs_2d(B: int):
    bs_locs = np.array([[0, 0], [0, -5], [0, 50]])
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
