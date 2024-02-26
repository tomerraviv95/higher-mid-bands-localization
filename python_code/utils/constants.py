from collections import namedtuple
from enum import Enum

import numpy as np

C = 300  # speed of light meter / micro-second
MU_SEC = 10 ** (-6)  # mu seconds factor
MEGA = 10 ** 6  # for the mega hertz frequencies
L_MAX = 4  # maximum number of paths for the synthetic channel
NF = 7  # noise figure in dB
N_0 = -174  # dBm
DEG = np.pi / 180  # conversion from degrees to pi
NS = 10  # number of pilot samples

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
    BEAMFORMER = 'BEAMSWEEPER'
    MUSIC = 'MUSIC'


class ScenarioType(Enum):
    SYNTHETIC = 'SYNTHETIC'
    NY = 'NY'


class BandType(Enum):
    SINGLE = 'SINGLE'
    MULTI = 'MULTI'
