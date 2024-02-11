from collections import namedtuple
from enum import Enum

C = 300  # speed of light meter / micro-second
MAX_DIST = 100  # maximum distance in meters supported in the simulation, but this could be smaller due to BW.
DATA_COEF = 10  # increase the data of the covariance matrix to make sure the covariance is enough
MU_SEC = 10 ** (-6)  # mu seconds factor
MEGA = 10 ** 6  # for the mega hertz frequencies
L_MAX = 4  # maximum number of paths for the synthetic channel
NF = 7  # noise figure in dB
N_0 = -174  # dBm

Channel = namedtuple("Channel", ["scatterers", "y", "bs", "AOA", "TOA", "ZOA", "band"])

Estimation = namedtuple("Estimation", ["AOA", "TOA", "ZOA", "POWER"], defaults=(None,) * 4)


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


class ScenarioType(Enum):
    SYNTHETIC = 'SYNTHETIC'
    NY = 'NY'


class BandType(Enum):
    SINGLE = 'SINGLE'
    MULTI = 'MULTI'
