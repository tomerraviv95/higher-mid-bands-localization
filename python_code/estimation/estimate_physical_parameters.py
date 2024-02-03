from typing import List

import numpy as np

from python_code import conf
from python_code.channel import get_channel
from python_code.estimation import Estimation
from python_code.estimation.angle import AngleEstimator3D, AngleEstimator2D
from python_code.estimation.angle_time import AngleTimeEstimator2D, AngleTimeEstimator3D
from python_code.estimation.estimations_combining import combine_estimations
from python_code.estimation.estimations_printer import printer_main
from python_code.estimation.time import TimeEstimator2D, TimeEstimator3D
from python_code.utils.bands_manipulation import Band
from python_code.utils.constants import EstimatorType, DimensionType, C

estimators = {
    EstimatorType.ANGLE: {DimensionType.Three.name: AngleEstimator3D, DimensionType.Two.name: AngleEstimator2D},
    EstimatorType.TIME: {DimensionType.Three.name: TimeEstimator3D, DimensionType.Two.name: TimeEstimator2D},
    EstimatorType.ANGLE_TIME: {DimensionType.Three.name: AngleTimeEstimator3D,
                               DimensionType.Two.name: AngleTimeEstimator2D}}


def estimate_physical_parameters(ue_pos: np.ndarray, bs_locs: np.ndarray, scatterers: np.ndarray,
                                 estimator_type: EstimatorType, bands: List[Band]) -> List[Estimation]:
    estimations = []
    # for each bs
    for i, bs_loc in enumerate(bs_locs):
        per_band_estimations = []
        # for each frequency sub-band
        for j, band in enumerate(bands):
            # generate the channel
            bs_ue_channel = get_channel(bs_loc, ue_pos, scatterers, band)
            if j == 0:
                print(f"Distance to user {bs_ue_channel.TOA[0] * C}[m], "
                      f"TOA[us]: {round(bs_ue_channel.TOA[0], 3)}, "
                      f"AOA to user {round(bs_ue_channel.AOA[0], 3)}[rad]")
            # choose the estimator based on the desired type
            estimator = estimators[estimator_type][conf.dimensions](band)
            # estimate delay / AOA / ZOA parameters for the current bs
            estimation = estimator.estimate(bs_ue_channel.y)
            per_band_estimations.append(estimation)
        # combine the estimations from the different bands together
        estimation = combine_estimations(per_band_estimations, bands, estimator_type)
        estimations.append(estimation)
        print(f"BS #{i} - {bs_loc}")
        printer_main(bs_ue_channel, estimation, estimator, estimator_type)
    return estimations
