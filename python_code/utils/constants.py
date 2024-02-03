from collections import namedtuple
from enum import Enum

C = 300  # speed of light meter / micro-second
MAX_DIST = 100  # maximum distance in meters supported in the simulation, but this could be smaller due to BW.
DATA_COEF = 2  # increase the data of the covariance matrix to make sure the covariance is enough
P_0 = 10 ** 4  # initial transmission power in watt

Channel = namedtuple("Channel", ["scatterers", "y", "AOA", "TOA", "ZOA", "band"])


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
