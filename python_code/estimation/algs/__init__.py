from python_code.estimation.algs.beamformer import Beamformer
from python_code.estimation.algs.multiband_beamforming import MultiBandBeamformer
from python_code.utils.constants import AlgType, BandType

ALGS_DICT = {AlgType.BEAMFORMER: {BandType.SINGLE: Beamformer, BandType.MULTI: MultiBandBeamformer}}
ALG_TYPE = AlgType.BEAMFORMER
