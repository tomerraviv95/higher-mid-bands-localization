from collections import namedtuple

from python_code.estimation.beam_sweeper import BeamSweeper
from python_code.estimation.music import MUSIC
from python_code.utils.constants import AlgType

ALGS_DICT = {AlgType.BEAMSWEEPER: BeamSweeper(2), AlgType.MUSIC: MUSIC(1.4)}
ALG_TYPE = AlgType.MUSIC

Estimation = namedtuple("Estimation", ["AOA", "TOA", "ZOA"], defaults=(None,) * 3)
