from typing import List

import numpy as np

from python_code import conf
from python_code.optimization.position_optimizer_2d import optimize_to_estimate_position_2d
from python_code.optimization.position_optimizer_3d import optimize_to_estimate_position_3d
from python_code.utils.constants import DimensionType, Estimation


def optimize_to_estimate_position(bs_locs: List[np.ndarray], estimations: List[Estimation]) -> np.ndarray:
    if conf.dimensions == DimensionType.Two.name:
        est_ue_pos = optimize_to_estimate_position_2d(np.array(bs_locs), estimations)
    else:
        est_ue_pos = optimize_to_estimate_position_3d(np.array(bs_locs), estimations)
    return est_ue_pos
