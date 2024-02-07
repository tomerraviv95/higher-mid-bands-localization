import numpy as np

from python_code import conf
from python_code.channel.synthetic_channel import get_synthetic_channel
from python_code.channel.synthetic_channel.bs_scatterers import create_bs_locs_2d, create_scatter_points_2d
from python_code.utils.bands_manipulation import Band
from python_code.utils.constants import ScenarioType, Channel


def get_channel(bs_ind: int, ue_pos: np.ndarray, band: Band) -> Channel:
    if conf.scenario == ScenarioType.SYNTHETIC.name:
        bs_ue_channel = get_synthetic_channel(bs_ind, ue_pos, band)
    elif conf.scenario == ScenarioType.NY.name:
        raise ValueError("Not implemented yet!!!")
    else:
        raise ValueError("Scenario is not implemented!!!")
    return bs_ue_channel
