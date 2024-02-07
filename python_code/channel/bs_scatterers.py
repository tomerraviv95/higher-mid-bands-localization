import numpy as np

from python_code import conf
from python_code.utils.constants import DimensionType


def create_scatter_points_2d(L: int) -> np.ndarray:
    scatterers = np.array([[5, 12], [5, 14], [2, 14]])
    return scatterers[:L - 1]


def create_scatter_points_3d(L: int) -> np.ndarray:
    scatterers = np.array([[15, 12, 3], [15, 14, 4], [12, 14, 8]])
    return scatterers[:L - 1]


def create_scatter_points(L: int) -> np.ndarray:
    assert L > 0
    create_scatter_funcs = {DimensionType.Three.name: create_scatter_points_3d(L),
                            DimensionType.Two.name: create_scatter_points_2d(L)}
    return create_scatter_funcs[conf.dimensions]


def create_bs_locs_2d(B: int) -> np.ndarray:
    bs_locs = np.array([[0, 0], [0, -5], [0, 5]])
    return bs_locs[:B]


def create_bs_locs_3d(B: int) -> np.ndarray:
    bs_locs = np.array([[0, 0, 10], [0, -5, 10], [0, 5, 10]])
    return bs_locs[:B]


def create_bs_locs(B: int) -> np.ndarray:
    assert B > 0
    create_bs_funcs = {DimensionType.Three.name: create_bs_locs_3d(B),
                       DimensionType.Two.name: create_bs_locs_2d(B)}
    return create_bs_funcs[conf.dimensions]
