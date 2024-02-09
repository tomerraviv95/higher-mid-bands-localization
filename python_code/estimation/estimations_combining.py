from typing import List

import numpy as np

from python_code.utils.bands_manipulation import Band
from python_code.utils.constants import EstimatorType, Estimation


def combine_estimations(estimations: List[Estimation], bands: List[Band], estimator_type: EstimatorType) -> Estimation:
    if len(estimations) == 1:
        return estimations[0]
    POWER = None
    if estimator_type == EstimatorType.ANGLE:
        # take the angle of the band with the highest number of antennas
        max_antennas_band_ind = np.argmax(np.array([band.Nr_x * band.Nr_y for band in bands]))
        AOA = estimations[max_antennas_band_ind].AOA
        ZOA = estimations[max_antennas_band_ind].ZOA
    else:
        AOA, ZOA = None, None

    if estimator_type == EstimatorType.TIME:
        # take the time part only
        max_time_separation_band_ind = np.argmax(np.array([band.BW / band.K for band in bands]))
        TOA = estimations[max_time_separation_band_ind].TOA
    else:
        TOA = None

    if estimator_type == EstimatorType.ANGLE_TIME:
        # concat all the options
        options = []
        for estimation in estimations:
            if estimation.AOA is None:
                continue
            for i in range(len(estimation.AOA)):
                aoa = estimation.AOA[i]
                toa = estimation.TOA[i]
                power = estimation.POWER[i]
                if estimation.ZOA is not None:
                    zoa = estimation.ZOA[i]
                else:
                    zoa = None
                options.append((aoa, toa, power, zoa))
        # remove duplicates
        options = list(set(options))
        AOA, TOA, POWER, ZOA = zip(*options)
        AOA, TOA, POWER, ZOA = list(AOA), list(TOA), list(POWER), list(ZOA)

    return Estimation(AOA=AOA, TOA=TOA, POWER=POWER, ZOA=ZOA)
