from enum import Enum

C = 300  # speed of light meter / micro-second
MAX_DIST = 500  # maximum distance in meters supported in the simulation
P_0 = 10 ** 5  # initial transmission power in watt


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
