import numpy as np

from python_code import conf
from python_code.channel.generate_2d_channel import get_2d_channel
from python_code.channel.ny_channel.ny_channel_loader_2d import get_ny_channel
from python_code.channel.synthetic_channel import get_synthetic_channel
from python_code.channel.synthetic_channel.bs_scatterers import create_bs_locs_2d, create_scatter_points_2d
from python_code.utils.bands_manipulation import Band
from python_code.utils.constants import Channel, DimensionType


def get_channel(bs_ind: int, ue_pos: np.ndarray, band: Band) -> Channel:
    if conf.dimensions == DimensionType.Three.name:
        bs_ue_channel = get_synthetic_channel(bs_ind, ue_pos, band)
    elif conf.dimensions == DimensionType.Two.name:
        bs_ue_channel = get_2d_channel(bs_ind, ue_pos, band)
    else:
        raise ValueError("Scenario is not implemented!!!")
    return bs_ue_channel
