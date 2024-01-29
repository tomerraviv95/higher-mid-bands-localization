import numpy as np

from python_code import conf
from python_code.channel.channel_generator_2d import get_2d_channel
from python_code.channel.channel_generator_3d import get_3d_channel
from python_code.utils.constants import DimensionType


def get_channel(bs_loc, ue_pos, scatterers):
    if conf.dimensions == DimensionType.Three.name:
        bs_ue_channel = get_3d_channel(np.array(bs_loc), ue_pos, scatterers)
    else:
        bs_ue_channel = get_2d_channel(np.array(bs_loc), ue_pos, scatterers)
    return bs_ue_channel