import numpy as np

from python_code import conf
from python_code.utils.constants import DimensionType


def create_scatter_points_2d(L: int):
    scatterers = np.array([[15, 12], [15, 14], [12, 14]])
    return scatterers[:L - 1]


def create_scatter_points_3d(L: int):
    scatterers = np.array([[9, -5, 1], [10, -6, 3], [5, -5, 8]])
    return scatterers[:L - 1]


def create_scatter_points(L: int):
    assert L > 0
    create_scatter_funcs = {DimensionType.Three.name: create_scatter_points_3d(L),
                            DimensionType.Two.name: create_scatter_points_2d(L)}
    return create_scatter_funcs[conf.dimensions]


def create_bs_locs_2d(B: int):
    bs_locs = np.array([[0, 0], [0, -5], [0, 50]])
    return bs_locs[:B]


def create_bs_locs_3d(B: int):
    bs_locs = np.array([[0, 5, 10], [0, -5, 10], [0, 50, 10]])
    return bs_locs[:B]


def create_bs_locs(B: int):
    assert B > 0
    create_bs_funcs = {DimensionType.Three.name: create_bs_locs_3d(B),
                       DimensionType.Two.name: create_bs_locs_2d(B)}
    return create_bs_funcs[conf.dimensions]
