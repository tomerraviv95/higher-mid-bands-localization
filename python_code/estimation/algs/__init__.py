from python_code.estimation import BeamSweeper, MUSIC
from python_code.utils.constants import AlgType

ALGS_DICT = {AlgType.BEAMSWEEPER: BeamSweeper(2), AlgType.MUSIC: MUSIC(1.4)}
ALG_TYPE = AlgType.MUSIC