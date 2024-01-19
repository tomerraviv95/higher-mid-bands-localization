from collections import namedtuple

from python_code.estimation.algs.beam_sweeper import BeamSweeper
from python_code.estimation.algs.music import MUSIC

Estimation = namedtuple("Estimation", ["AOA", "TOA", "ZOA"], defaults=(None,) * 3)
