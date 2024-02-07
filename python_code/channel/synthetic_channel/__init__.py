import numpy as np

from python_code import conf
from python_code.channel.synthetic_channel.bs_scatterers import create_bs_locs_3d, create_scatter_points_3d, \
    create_bs_locs_2d, create_scatter_points_2d
from python_code.channel.synthetic_channel.channel_generator_2d import get_2d_basic_channel
from python_code.channel.synthetic_channel.channel_generator_3d import get_3d_basic_channel
from python_code.utils.bands_manipulation import Band
from python_code.utils.constants import DimensionType, Channel


def get_synthetic_channel(bs_ind: int, ue_pos: np.ndarray, band: Band) -> Channel:
    if conf.dimensions == DimensionType.Three.name:
        bs_loc = create_bs_locs_3d(bs_ind)
        scatterers = create_scatter_points_3d(conf.L)
        bs_ue_channel = get_3d_basic_channel(bs_loc, ue_pos, scatterers, band)
    else:
        bs_loc = create_bs_locs_2d(bs_ind)
        scatterers = create_scatter_points_2d(conf.L)
        bs_ue_channel = get_2d_basic_channel(bs_loc, ue_pos, scatterers, band)
    return bs_ue_channel
