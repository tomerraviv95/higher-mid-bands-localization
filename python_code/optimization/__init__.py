from python_code import conf
from python_code.optimization.position_optimizer_2d import optimize_to_estimate_position_2d
from python_code.optimization.position_optimizer_3d import optimize_to_estimate_position_3d
from python_code.utils.constants import DimensionType


def optimize_to_estimate_position(bs_locs, estimations):
    if conf.dimensions == DimensionType.Two.name:
        est_ue_pos = optimize_to_estimate_position_2d(bs_locs, estimations)
    else:
        est_ue_pos = optimize_to_estimate_position_3d(bs_locs, estimations)
    return est_ue_pos
