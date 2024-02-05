from typing import List, Tuple

import numpy as np

from python_code import conf
from python_code.channel import get_channel
from python_code.estimation import estimators
from python_code.estimation.estimations_combining import combine_estimations
from python_code.plotting.estimations_printer import printer_main
from python_code.plotting.plotter import print_channel
from python_code.utils.bands_manipulation import Band
from python_code.utils.constants import EstimatorType, Channel, Estimation, BandType

BANDS_ESTIMATION = BandType.MULTI


def estimate_physical_parameters(ue_pos: np.ndarray, bs_locs: np.ndarray, scatterers: np.ndarray,
                                 estimator_type: EstimatorType, bands: List[Band]) -> List[Estimation]:
    estimations = []
    # for each bs
    for i, bs_loc in enumerate(bs_locs):
        if conf.band_type == BandType.SINGLE.name:
            bs_ue_channel, estimation, estimator = separate_bands_estimator(bands, bs_loc, estimator_type, scatterers,
                                                                            ue_pos)
        elif conf.band_type == BandType.MULTI.name:
            bs_ue_channel, estimation, estimator = sim_bands_estimator(bands, bs_loc, estimator_type, scatterers,
                                                                       ue_pos)
        else:
            raise ValueError("No such band type for estimation, either single or multi!!")
        estimations.append(estimation)
        print(f"BS #{i} - {bs_loc}")
        printer_main(bs_ue_channel, estimation, estimator, estimator_type)
    return estimations


def separate_bands_estimator(bands: List[Band], bs_loc: np.ndarray, estimator_type: EstimatorType,
                             scatterers: np.ndarray, ue_pos: np.ndarray) -> Tuple[Channel, Estimation, object]:
    per_band_estimations = []
    # for each frequency sub-band
    for j, band in enumerate(bands):
        # generate the channel
        bs_ue_channel = get_channel(bs_loc, ue_pos, scatterers, band)
        if j == 0:
            print_channel(bs_ue_channel)
        # choose the estimator based on the desired type
        estimator = estimators[estimator_type][conf.dimensions]([band])
        # estimate delay / AOA / ZOA parameters_2d for the current bs
        estimation = estimator.estimate(bs_ue_channel.y)
        per_band_estimations.append(estimation)
    # combine the estimations from the different bands together
    estimation = combine_estimations(per_band_estimations, bands, estimator_type)
    return bs_ue_channel, estimation, estimator


def sim_bands_estimator(bands: List[Band], bs_loc: np.ndarray, estimator_type: EstimatorType,
                        scatterers: np.ndarray, ue_pos: np.ndarray) -> Tuple[Channel, Estimation, object]:
    per_band_y = []
    # for each frequency sub-band
    for j, band in enumerate(bands):
        # generate the channel
        bs_ue_channel = get_channel(bs_loc, ue_pos, scatterers, band)
        if j == 0:
            print_channel(bs_ue_channel)
        # append to list
        per_band_y.append(bs_ue_channel.y)
    # choose the estimator based on the desired type
    estimator = estimators[estimator_type][conf.dimensions](bands)
    # estimate delay / AOA / ZOA parameters_2d for the current bs based on all measurements from all bands
    estimation = estimator.estimate(per_band_y)
    return bs_ue_channel, estimation, estimator
