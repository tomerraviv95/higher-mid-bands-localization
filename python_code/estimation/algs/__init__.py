from python_code import conf
from python_code.estimation.algs.beamformer import Beamformer
from python_code.estimation.algs.multiband_beamforming import MultiBandBeamformer
from python_code.estimation.algs.music import MUSIC
from python_code.utils.constants import AlgType, BandType

# Following a factory design pattern
ALGS_DICT = {AlgType.BEAMFORMER: {BandType.SINGLE: Beamformer,
                                  BandType.MULTI: MultiBandBeamformer},
             AlgType.MUSIC: {BandType.SINGLE: MUSIC,
                             BandType.MULTI: MUSIC}}

ALG_TYPES = {'Beamformer': AlgType.BEAMFORMER, 'MUSIC': AlgType.MUSIC}
ALG_TYPE = ALG_TYPES[conf.alg]
