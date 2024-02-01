import math

import numpy as np

from python_code import conf

WALLS = np.array([[8, 8], [8, 12], [12, 12], [12, 8], [8, 8]])
LOSS_FACTOR = {6000: 1.1, 24000: 100}


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def calc_power(P0, bs_loc, ue_pos):
    for wall1, wall2 in zip(WALLS[:-1], WALLS[1:]):
        if intersect(bs_loc, ue_pos, wall1, wall2):
            P0 /= LOSS_FACTOR[conf.fc]
    return P0


def compute_path_loss(toa):
    loss_db = 20 * math.log10(toa) + 20 * math.log10(conf.fc) + 20 * math.log10(4 * math.pi)
    return 10 ** (loss_db / 20)
