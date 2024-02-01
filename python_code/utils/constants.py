from enum import Enum

import numpy as np

C = 300  # speed of light meter / micro-second
MAX_DIST = 100  # maximum distance in meters supported in the simulation, but this could be smaller due to BW.
P_0 = 10 ** 5  # initial transmission power in watt
INITIAL_POWER = P_0 * np.sqrt(1 / 2) * (np.random.randn(1) + np.random.randn(1) * 1j)
DATA_COEF = 3

class EstimatorType(Enum):
    ANGLE = 'ANGLE'
    TIME = 'TIME'
    ANGLE_TIME = 'ANGLE_TIME'


class ChannelBWType(Enum):
    NARROWBAND = 'NARROWBAND'
    WIDEBAND = 'WIDEBAND'


class AlgType(Enum):
    CAPON = 'BEAMSWEEPER'
    MUSIC = 'MUSIC'


class DimensionType(Enum):
    Two = 'Two'
    Three = 'Three'
