from python_code.estimation.algs.capon_beamforming import CaponBeamforming
from python_code.estimation.algs.music import MUSIC
from python_code.utils.constants import AlgType

ALGS_DICT = {AlgType.CAPON: CaponBeamforming(10), AlgType.MUSIC: MUSIC(1.4)}
ALG_TYPE = AlgType.CAPON