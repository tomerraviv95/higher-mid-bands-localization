from python_code.estimation.algs.capon_beamforming import CaponBeamforming
from python_code.estimation.algs.multiband_capon_beamforming import MultiBandCaponBeamforming
from python_code.estimation.algs.music import MUSIC
from python_code.utils.constants import AlgType, BandType

ALGS_DICT = {AlgType.CAPON: {BandType.SINGLE: CaponBeamforming, BandType.MULTI: MultiBandCaponBeamforming},
             AlgType.MUSIC: {BandType.SINGLE: MUSIC}}
ALG_TYPE = AlgType.CAPON
