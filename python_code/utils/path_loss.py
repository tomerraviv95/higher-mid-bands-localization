import math
from typing import Tuple

import numpy as np

##### this part is for the synthetic channel only #####


WALLS = np.array([[8, 11], [8, 15], [11, 15], [11, 11], [8, 11]])
LOSS_FACTOR = {6000: 1.1, 24000: 100}


def ccw(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


# Return true if line segments AB and CD intersect
def intersect(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float], d: Tuple[float, float]) -> bool:
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def calc_power(P0: float, bs_loc: np.ndarray, ue_pos: np.ndarray, fc: float) -> float:
    # decrease the power if the path propagates through a wall
    for wall1, wall2 in zip(WALLS[:-1], WALLS[1:]):
        if intersect(bs_loc, ue_pos, wall1, wall2):
            P0 /= LOSS_FACTOR[fc]
    return P0


##### this part is for the synthetic channel only #####

def compute_path_loss(toa: float, fc: float) -> float:
    # free path loss computation
    loss_db = 20 * math.log10(toa) + 20 * math.log10(fc) + 20 * math.log10(4 * math.pi)
    return 10 ** (loss_db / 20)


def watt_power_from_dbm(dbm_power: float) -> float:
    return 10 ** (dbm_power / 20)
