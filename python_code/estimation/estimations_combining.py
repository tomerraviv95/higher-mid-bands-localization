from typing import List

import numpy as np

from python_code.estimation import Estimation
from python_code.utils.bands_manipulation import Band
from python_code.utils.constants import EstimatorType


def combine_estimations(estimations: List[Estimation], bands: List[Band], estimator_type: EstimatorType) -> Estimation:
    if len(estimations) == 1:
        return estimations[0]

    if estimator_type == EstimatorType.ANGLE:
        # take the angle of the band with the highest number of antennas
        max_antennas_band_ind = np.argmax(np.array([band.Nr_x * band.Nr_y for band in bands]))
        AOA = estimations[max_antennas_band_ind].AOA
        ZOA = estimations[max_antennas_band_ind].ZOA
    else:
        AOA, ZOA = None, None

    if estimator_type == EstimatorType.TIME:
        # take the time
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
                if estimation.ZOA:
                    zoa = estimation.ZOA[i]
                else:
                    zoa = None
                options.append((aoa, toa, zoa))
        # remove duplicates
        options = list(set(options))
        AOA, TOA, ZOA = zip(*options)
        AOA, TOA, ZOA = list(AOA), list(TOA), list(ZOA)

    return Estimation(AOA=AOA, TOA=TOA, ZOA=ZOA)