from enum import Enum

C = 300  # speed of light meter / us


class EstimatorType(Enum):
    ANGLE = 'ANGLE'
    TIME = 'TIME'
    ANGLE_TIME = 'ANGLE_TIME'


class ChannelBWType(Enum):
    NARROWBAND = 'NARROWBAND'
    WIDEBAND = 'WIDEBAND'


class AlgType(Enum):
    BEAMSWEEPER = 'BEAMSWEEPER'
    MUSIC = 'MUSIC'


class DimensionType(Enum):
    Two = 'Two'
    Three = 'Three'
